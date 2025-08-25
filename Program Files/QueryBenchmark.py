#!/usr/bin/env python3
"""
Performance Benchmarking Suite for QB4OLAP Mental Health Analytics
Addiction Nexus: Multidimensional Performance Evaluation Framework

Enhancements (2025-08-20):
- Separate 'slice' and 'dice' queries (replacing combined slice_dice).
- Save observation tables as images for every query run
  to observation_plots/<query>_observations__size<...>__iter<...>.png
- Preserve existing functionality (roll_up, drill_down, data_integrity,
  synthetic generation hooks, reporting & plots).
"""

import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from faker import Faker
from SPARQLWrapper import SPARQLWrapper, JSON, POST
import warnings
warnings.filterwarnings('ignore')

# ---------------- Logging ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('benchmark_performance.log'), logging.StreamHandler()]
)


class MentalHealthBenchmark:
    """
    Comprehensive benchmarking suite for QB4OLAP mental health analytics
    """

    def __init__(self, sparql_endpoint: str = "http://localhost:8890/sparql"):
        self.sparql_endpoint = sparql_endpoint
        self.sparql = SPARQLWrapper(sparql_endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setMethod(POST)

        # Initialize Faker for synthetic data generation
        self.fake = Faker()

        # Benchmark results storage
        self.benchmark_results: Dict[str, list] = {
            'dataset_sizes': [],
            'query_types': [],
            'response_times': [],
            'record_counts': [],
            'execution_dates': []
        }

        # Ensure output dirs exist
        os.makedirs("observation_plots", exist_ok=True)
        os.makedirs("benchmark_plots", exist_ok=True)

        # ---------------- SPARQL Query Templates ---------------- #
        self.queries: Dict[str, str] = {
            # ROLL-UP: average stress (ordinalized) at AgeGroup level
            'roll_up': """
            PREFIX adnx: <https://purl.archive.org/addiction-nexus#>
            PREFIX qb4o: <http://purl.org/qb4olap/cubes#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

            SELECT ?ageGroup (COUNT(?obs) AS ?obsCount) (AVG(?stressLevelOrder) AS ?avgStressLevel)
            WHERE {
              GRAPH <http://localhost:8890/ton1> {
                ?obs a qb4o:Observation ;
                     qb4o:inCube adnx:MentalHealthCube ;
                     adnx:ageDimension ?ageUri ;
                     adnx:stressLevelDimension ?stressUri .
                ?ageUri qb4o:rollsUpTo ?ageGroupUri .
                ?ageGroupUri qb4o:memberOf adnx:AgeGroupLevel ;
                            skos:prefLabel ?ageGroup .
                ?stressUri skos:prefLabel ?stressValue .
                BIND(
                  IF(CONTAINS(LCASE(?stressValue), "low"), 1,
                     IF(CONTAINS(LCASE(?stressValue), "medium"), 2,
                        IF(CONTAINS(LCASE(?stressValue), "high"), 3, 0))) AS ?stressLevelOrder
                )
              }
            }
            GROUP BY ?ageGroup
            HAVING (COUNT(?obs) > 10)
            ORDER BY DESC(?avgStressLevel)
            """,

            # DRILL-DOWN: smokers by country & occupation
            'drill_down': """
            PREFIX adnx: <https://purl.archive.org/addiction-nexus#>
            PREFIX qb4o: <http://purl.org/qb4olap/cubes#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

            SELECT ?country ?occupation (COUNT(?obs) AS ?smokerCount)
            WHERE {
              GRAPH <http://localhost:8890/ton1> {
                ?obs a qb4o:Observation ;
                     qb4o:inCube adnx:MentalHealthCube ;
                     adnx:countryDimension ?countryUri ;
                     adnx:occupationDimension ?occupationUri ;
                     adnx:observationHasSmokingHabit ?smokingUri .
                ?countryUri skos:prefLabel ?country .
                ?occupationUri skos:prefLabel ?occupation .
                ?smokingUri skos:prefLabel ?smokingHabit .
                FILTER (CONTAINS(LCASE(?smokingHabit), "smoker") && !CONTAINS(LCASE(?smokingHabit), "non"))
              }
            }
            GROUP BY ?country ?occupation
            HAVING (COUNT(?obs) > 5)
            ORDER BY ?country ?occupation
            """,

            # SLICE: single-dimension filter (stress = high), avg work hours by occupation
            'slice': """
            PREFIX adnx: <https://purl.archive.org/addiction-nexus#>
            PREFIX qb4o: <http://purl.org/qb4olap/cubes#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            SELECT ?occupation (AVG(?workHours) AS ?avgWorkHours) (COUNT(?obs) AS ?obsCount)
            WHERE {
              GRAPH <http://localhost:8890/ton1> {
                ?obs a qb4o:Observation ;
                     qb4o:inCube adnx:MentalHealthCube ;
                     adnx:stressLevelDimension ?stressUri ;
                     adnx:occupationDimension ?occupationUri ;
                     adnx:workHoursMeasure ?workHours .
                ?stressUri skos:prefLabel ?stressValue .
                ?occupationUri skos:prefLabel ?occupation .
                FILTER (CONTAINS(LCASE(?stressValue), "high"))
              }
            }
            GROUP BY ?occupation
            ORDER BY DESC(?avgWorkHours)
            """,

            # DICE: multi-dimension filter (country ∈ {AU, IN}, severity = medium), avg alcohol by gender
            'dice': """
            PREFIX adnx: <https://purl.archive.org/addiction-nexus#>
            PREFIX qb4o: <http://purl.org/qb4olap/cubes#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            SELECT ?gender (AVG(?alcoholOrder) AS ?avgAlcohol) (COUNT(?obs) AS ?obsCount)
            WHERE {
              GRAPH <http://localhost:8890/ton1> {
                ?obs a qb4o:Observation ;
                     qb4o:inCube adnx:MentalHealthCube ;
                     adnx:countryDimension ?countryUri ;
                     adnx:severityDimension ?severityUri ;
                     adnx:genderDimension ?genderUri ;
                     adnx:observationHasAlcoholConsumption ?alcoholUri .
                ?countryUri skos:prefLabel ?country .
                ?severityUri skos:prefLabel ?severity .
                ?genderUri skos:prefLabel ?gender .
                ?alcoholUri skos:prefLabel ?alcoholValue .
                BIND(
                  IF(CONTAINS(LCASE(?alcoholValue), "non"), 0,
                     IF(CONTAINS(LCASE(?alcoholValue), "social"), 1,
                        IF(CONTAINS(LCASE(?alcoholValue), "regular"), 2,
                           IF(CONTAINS(LCASE(?alcoholValue), "heavy"), 3, 0)))) AS ?alcoholOrder
                )
                FILTER (LCASE(?country) IN ("australia", "india") && CONTAINS(LCASE(?severity), "medium"))
              }
            }
            GROUP BY ?gender
            ORDER BY ?gender
            """,

            # DATA INTEGRITY: simple count of observations in the cube
            'data_integrity': """
            PREFIX adnx: <https://purl.archive.org/addiction-nexus#>
            PREFIX qb4o: <http://purl.org/qb4olap/cubes#>
            SELECT (COUNT(?obs) AS ?totalObservations)
            WHERE {
              GRAPH <http://localhost:8890/ton1> {
                ?obs a qb4o:Observation ;
                     qb4o:inCube adnx:MentalHealthCube .
              }
            }
            """
        }

    # ---------------- Synthetic Data (optional hook) ---------------- #
    def generate_synthetic_data(self, target_size: int, base_data_path: str) -> str:
        logging.info(f"Generating synthetic dataset with {target_size} records...")
        base_df = pd.read_csv(base_data_path)  # schema anchor; not used below explicitly

        countries = ['USA', 'Canada', 'Australia', 'UK', 'Germany', 'India', 'Other']
        occupations = ['IT', 'Healthcare', 'Education', 'Engineering', 'Finance', 'Sales', 'Other']
        genders = ['Male', 'Female', 'Non-binary', 'Prefer not to say']
        severities = ['None', 'Low', 'Medium', 'High']
        stress_levels = ['Low', 'Medium', 'High']
        smoking_habits = ['Non-Smoker', 'Occasional Smoker', 'Regular Smoker', 'Heavy Smoker']
        alcohol_consumption = ['Non-Drinker', 'Social Drinker', 'Regular Drinker', 'Heavy Drinker']
        diet_qualities = ['Healthy', 'Average', 'Unhealthy']

        synthetic_data = []
        for i in range(target_size):
            age = np.random.normal(42, 15)
            age = max(18, min(65, int(age)))
            rec = {
                'User_ID': i + 1,
                'Age': age,
                'Gender': np.random.choice(genders),
                'Occupation': np.random.choice(occupations),
                'Country': np.random.choice(countries),
                'Mental_Health_Condition': np.random.choice(['Yes', 'No'], p=[0.3, 0.7]),
                'Severity': np.random.choice(severities, p=[0.4, 0.2, 0.2, 0.2]),
                'Consultation_History': np.random.choice(['Yes', 'No'], p=[0.4, 0.6]),
                'Stress_Level': np.random.choice(stress_levels, p=[0.3, 0.4, 0.3]),
                'Sleep_Hours': round(np.random.normal(7, 1.5), 1),
                'Work_Hours': int(np.random.normal(45, 15)),
                'Physical_Activity_Hours': int(np.random.exponential(3)),
                'Social_Media_Usage': round(np.random.exponential(2), 1),
                'Diet_Quality': np.random.choice(diet_qualities, p=[0.3, 0.4, 0.3]),
                'Smoking_Habit': np.random.choice(smoking_habits, p=[0.4, 0.2, 0.2, 0.2]),
                'Alcohol_Consumption': np.random.choice(alcohol_consumption, p=[0.3, 0.3, 0.2, 0.2]),
                'Medication_Usage': np.random.choice(['Yes', 'No'], p=[0.3, 0.7])
            }
            rec['Sleep_Hours'] = max(3, min(12, rec['Sleep_Hours']))
            rec['Work_Hours'] = max(20, min(80, rec['Work_Hours']))
            rec['Physical_Activity_Hours'] = min(10, rec['Physical_Activity_Hours'])
            rec['Social_Media_Usage'] = min(8, rec['Social_Media_Usage'])
            synthetic_data.append(rec)

        out = f'mental_health_synthetic_{target_size}.csv'
        pd.DataFrame(synthetic_data).to_csv(out, index=False)
        logging.info(f"Synthetic dataset saved to {out}")
        return out

    # ---------------- Core Execution ---------------- #
    def execute_sparql_query(
        self,
        query: str,
        query_type: str,
        save_observation: bool = True,
        filename_suffix: str = ""
    ) -> Tuple[float, int, Optional[Dict]]:
        """
        Execute SPARQL query, time it, optionally save observations image.

        Returns:
            (execution_time_seconds, result_count, results_json_or_None)
        """
        try:
            self.sparql.setQuery(query)
            t0 = time.time()
            results = self.sparql.query().convert()
            t1 = time.time()

            exec_time = t1 - t0
            bindings = results.get("results", {}).get("bindings", [])
            result_count = len(bindings)

            logging.info(f"{query_type} executed in {exec_time:.3f}s, results={result_count}")

            if save_observation:
                try:
                    self.save_observations(query_type, results, filename_suffix)
                except Exception as e:
                    logging.warning(f"save_observations failed for {query_type}: {e}")

            return exec_time, result_count, results

        except Exception as e:
            logging.error(f"Query execution failed for {query_type}: {e}")
            return -1.0, 0, None

    # ---------------- Observation Table Saver ---------------- #
    def save_observations(self, query_type: str, results: Dict, suffix: str = "") -> None:
        """
        Render query result rows into a simple table image using matplotlib.
        Files saved under observation_plots/.
        """
        bindings = (results or {}).get("results", {}).get("bindings", [])
        safe_suffix = f"__{suffix}" if suffix else ""
        out_path = os.path.join("observation_plots", f"{query_type}_observations{safe_suffix}.png")

        # Map query types to column extraction logic
        # Helper to safely pull a value
        def val(row, key): return row.get(key, {}).get('value', '')

        rows = []
        columns = []

        if query_type == 'roll_up':
            columns = ['Age Group', 'Obs Count', 'Avg Stress Level']
            for r in bindings:
                rows.append([val(r, 'ageGroup'), val(r, 'obsCount'), val(r, 'avgStressLevel')])

        elif query_type == 'drill_down':
            columns = ['Country', 'Occupation', 'Smoker Count']
            for r in bindings:
                rows.append([val(r, 'country'), val(r, 'occupation'), val(r, 'smokerCount')])

        elif query_type == 'slice':
            columns = ['Occupation', 'Avg Work Hours', 'Obs Count']
            for r in bindings:
                rows.append([val(r, 'occupation'), val(r, 'avgWorkHours'), val(r, 'obsCount')])

        elif query_type == 'dice':
            columns = ['Gender', 'Avg Alcohol', 'Obs Count']
            for r in bindings:
                rows.append([val(r, 'gender'), val(r, 'avgAlcohol'), val(r, 'obsCount')])

        elif query_type == 'data_integrity':
            columns = ['Total Observations']
            total = val(bindings[0], 'totalObservations') if bindings else "0"
            rows = [[total]]

        else:
            # Generic fallback – dump whatever keys appear
            keys = sorted({k for r in bindings for k in r.keys()})
            columns = keys
            for r in bindings:
                rows.append([val(r, k) for k in keys])

        # If no data, show a single "No results" row
        if not rows:
            columns = columns or ['Info']
            rows = [["No results"]]

        # --- Draw table ---
        plt.figure(figsize=(10, 0.6 + 0.4 * max(1, len(rows))))
        plt.axis('off')
        title = f'{query_type.replace("_", " ").title()} Observations'
        plt.title(title, fontsize=13, fontweight='bold', pad=12)

        table = plt.table(
            cellText=rows,
            colLabels=columns,
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)

        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved observations: {out_path}")

    # ---------------- Benchmark Suite ---------------- #
    def run_benchmark_suite(self, dataset_sizes: List[int], iterations: int = 10):
        """
        Execute benchmark suite across dataset sizes.
        Note: This simulates by running queries repeatedly; actual scaling
        requires loading different sized graphs into Virtuoso.
        """
        logging.info(f"Starting benchmark suite with {len(dataset_sizes)} dataset sizes")
        for size in dataset_sizes:
            logging.info(f"--- Dataset size = {size} ---")

            for query_name, query in self.queries.items():
                if query_name == 'data_integrity':
                    # Skip in main loop; you can call it separately if needed
                    continue

                times = []

                for it in range(iterations):
                    suffix = f"size{size}__iter{it+1}"
                    exec_time, result_count, _ = self.execute_sparql_query(
                        query=query,
                        query_type=query_name,
                        save_observation=True,
                        filename_suffix=suffix
                    )

                    if exec_time >= 0:
                        times.append(exec_time)
                        self.benchmark_results['dataset_sizes'].append(size)
                        self.benchmark_results['query_types'].append(query_name)
                        self.benchmark_results['response_times'].append(exec_time)
                        self.benchmark_results['record_counts'].append(result_count)
                        self.benchmark_results['execution_dates'].append(datetime.now())

                if times:
                    logging.info(
                        f"  {query_name}: avg={np.mean(times):.3f}s, std={np.std(times):.3f}s, "
                        f"min={np.min(times):.3f}s, max={np.max(times):.3f}s"
                    )

    # ---------------- Integrity Check ---------------- #
    def validate_data_integrity(self, expected_count: int) -> bool:
        exec_time, _count, results = self.execute_sparql_query(
            self.queries['data_integrity'], 'data_integrity',
            save_observation=True, filename_suffix="integrity"
        )
        bindings = (results or {}).get("results", {}).get("bindings", [])
        actual = int(bindings[0]['totalObservations']['value']) if bindings else 0
        ok = (actual == expected_count)
        logging.info(f"Data integrity: expected={expected_count}, actual={actual}, ok={ok}")
        return ok

    # ---------------- Reporting & Plots ---------------- #
    def generate_performance_report(self) -> pd.DataFrame:
        df = pd.DataFrame(self.benchmark_results)
        if df.empty:
            logging.warning("No benchmark results to report.")
            return pd.DataFrame()

        summary = (
            df.groupby(['dataset_sizes', 'query_types'])['response_times']
              .agg(['count', 'mean', 'std', 'min', 'max', 'median'])
              .round(4)
              .reset_index()
              .rename(columns={
                  'count': 'iterations',
                  'mean': 'avg_time',
                  'std': 'std_time',
                  'min': 'min_time',
                  'max': 'max_time',
                  'median': 'median_time'
              })
        )
        logging.info("Performance report generated.")
        return summary

    def create_visualizations(self, results_df: pd.DataFrame, save_path: str = "benchmark_plots"):
        os.makedirs(save_path, exist_ok=True)

       # 1) Scalability curves: avg_time vs dataset size per query type (in milliseconds)
        plt.figure(figsize=(12, 8))
        for q in results_df['query_types'].unique():
            sub = results_df[results_df['query_types'] == q]
            series = sub.groupby('dataset_sizes')['avg_time'].mean() * 1000  # Convert to ms
            plt.plot(series.index, series.values, marker='o', linewidth=2, label=q.replace('_', ' ').title())

        plt.xlabel('Dataset Size (records)', fontsize=12)
        plt.ylabel('Average Response Time (ms)', fontsize=12)
        plt.title('OLAP Query Performance Scalability Analysis (ms)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'{save_path}/scalability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2) Box plot over raw execution times
        raw = pd.DataFrame(self.benchmark_results)
        if not raw.empty:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=raw, x='query_types', y='response_times')
            plt.xlabel('Query Type', fontsize=12)
            plt.ylabel('Response Time (seconds)', fontsize=12)
            plt.title('Response Time Distribution by Query Type', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{save_path}/response_time_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 3) Heatmap of avg_time by query vs dataset size
        pivot = results_df.pivot_table(values='avg_time', index='query_types', columns='dataset_sizes', aggfunc='mean')
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title('Performance Heatmap: Average Response Time (seconds)', fontsize=14, fontweight='bold')
        plt.xlabel('Dataset Size', fontsize=12)
        plt.ylabel('Query Type', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{save_path}/performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def export_results(self, filename: str = "benchmark_results.json"):
        export_data = self.benchmark_results.copy()
        export_data['execution_dates'] = [dt.isoformat() for dt in export_data['execution_dates']]
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        logging.info(f"Benchmark results exported to {filename}")


# ---------------- Main ---------------- #
def main():
    benchmark = MentalHealthBenchmark()

    # Parameterize as needed
    dataset_sizes = [10000, 50000, 200000]  # align with paper scalability claims
    iterations = 10

    print("=" * 60)
    print("QB4OLAP Mental Health Analytics - Performance Benchmark")
    print("=" * 60)

    # Run suite
    benchmark.run_benchmark_suite(dataset_sizes, iterations=iterations)

    # Report
    report = benchmark.generate_performance_report()
    if not report.empty:
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY REPORT")
        print("=" * 60)
        print(report.to_string(index=False))

        # Visualizations (scalability, boxplot, heatmap)
        benchmark.create_visualizations(report)

        # Export raw results (for paper tables, e.g., section 4.2.1)
        benchmark.export_results()

    # Optional integrity check (adjust expected_count to match your load)
    # benchmark.validate_data_integrity(expected_count=50000)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE - Results saved to benchmark_results.json")
    print("Observation images saved in: observation_plots/")
    print("Benchmark plots saved in:   benchmark_plots/")
    print("=" * 60)


if __name__ == "__main__":
    main()
