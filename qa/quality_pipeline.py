""" Automated Quality Assurance Pipeline
Author: Member 5
"""

import subprocess
import os
import json
import yaml
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import git


@dataclass
class QualityGate:
    """Quality gate definition"""
    name: str
    threshold: float
    current_value: float
    status: str
    details: Dict[str, Any]


@dataclass
class PipelineStage:
    """Pipeline stage definition"""
    name: str
    commands: List[str]
    timeout: int
    required: bool
    parallel: bool = False


class QualityPipeline:
    """Automated quality assurance pipeline"""

    def __init__(self, config_path: str = 'qa/pipeline_config.yaml'):
        self.config_path = config_path
        self.config = self.load_configuration()
        self.quality_gates = []
        self.pipeline_results = {}
        self.setup_logging()

        # Initialize Git repository info
        try:
            self.repo = git.Repo('.')
            self.current_commit = self.repo.head.commit.hexsha
            self.current_branch = self.repo.active_branch.name
        except Exception:
            self.repo = None
            self.current_commit = 'unknown'
            self.current_branch = 'unknown'

        print("Quality Pipeline initialized")

    def load_configuration(self) -> Dict:
        """Load pipeline configuration"""
        default_config = {
            'quality_gates': {
                'code_coverage': {'threshold': 80.0, 'enabled': True},
                'test_success_rate': {'threshold': 95.0, 'enabled': True},
                'performance_score': {'threshold': 85.0, 'enabled': True},
                'security_score': {'threshold': 90.0, 'enabled': True},
                'code_quality_score': {'threshold': 3.5, 'enabled': True}
            },
            'stages': [
                {
                    'name': 'Code Quality Analysis',
                    'commands': [
                        'flake8 --max-line-length=120 --count --statistics .',
                        'pylint src/ --output-format=json --reports=yes',
                        'bandit -r src/ -f json'
                    ],
                    'timeout': 300,
                    'required': True
                },
                {
                    'name': 'Unit Tests',
                    'commands': [
                        'python -m pytest tests/unit/ -v --tb=short --junitxml=tests/results/unit_tests.xml',
                        'python -m coverage run -m pytest tests/unit/',
                        'python -m coverage report --format=json'
                    ],
                    'timeout': 600,
                    'required': True
                },
                {
                    'name': 'Integration Tests',
                    'commands': [
                        'python -m pytest tests/integration/ -v --tb=short --junitxml=tests/results/integration_tests.xml'
                    ],
                    'timeout': 900,
                    'required': True
                },
                {
                    'name': 'Performance Tests',
                    'commands': [
                        'python tests/performance_suite.py'
                    ],
                    'timeout': 1200,
                    'required': False
                },
                {
                    'name': 'Security Scan',
                    'commands': [
                        'safety check --json',
                        'bandit -r . -f json -o tests/results/security_scan.json'
                    ],
                    'timeout': 300,
                    'required': True
                }
            ],
            'notifications': {
                'slack_webhook': None,
                'email_recipients': [],
                'teams_webhook': None
            },
            'artifacts': {
                'retention_days': 30,
                'output_directory': 'tests/results'
            }
        }
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # Merge with defaults
                    default_config.update(user_config)
            except Exception as e:
                print(f"Error loading config: {e}, using defaults")
        return default_config

    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path('tests/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'quality_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def run_pipeline(self) -> Dict:
        """Run the complete quality pipeline"""
        start_time = time.time()
        pipeline_result = {
            'timestamp': time.time(),
            'commit': self.current_commit,
            'branch': self.current_branch,
            'stages': {},
            'quality_gates': {},
            'overall_status': 'pending',
            'execution_time': 0,
            'artifacts': []
        }
        self.logger.info("Starting Quality Pipeline execution")
        self.logger.info(f"Commit: {self.current_commit}")
        self.logger.info(f"Branch: {self.current_branch}")
        try:
            # Create results directory
            results_dir = Path(self.config['artifacts']['output_directory'])
            results_dir.mkdir(parents=True, exist_ok=True)

            # Run each stage
            for stage_config in self.config['stages']:
                stage = PipelineStage(**stage_config)
                stage_result = self.run_stage(stage)
                pipeline_result['stages'][stage.name] = stage_result

                # Stop if required stage fails
                if stage.required and stage_result['status'] == 'failed':
                    self.logger.error(f"Required stage '{stage.name}' failed, stopping pipeline")
                    pipeline_result['overall_status'] = 'failed'
                    break

            # Evaluate quality gates
            if pipeline_result['overall_status'] != 'failed':
                quality_gate_results = self.evaluate_quality_gates()
                pipeline_result['quality_gates'] = quality_gate_results

                # Determine overall status
                if all(gate['status'] == 'passed' for gate in quality_gate_results.values()):
                    pipeline_result['overall_status'] = 'passed'
                else:
                    pipeline_result['overall_status'] = 'failed'

            # Calculate execution time
            pipeline_result['execution_time'] = time.time() - start_time

            # Generate artifacts
            self.generate_artifacts(pipeline_result)

            # Send notifications
            self.send_notifications(pipeline_result)
            self.logger.info(f"Pipeline completed with status: {pipeline_result['overall_status']}")
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            pipeline_result['overall_status'] = 'error'
            pipeline_result['error'] = str(e)
        return pipeline_result

    def run_stage(self, stage: PipelineStage) -> Dict:
        """Run a single pipeline stage"""
        self.logger.info(f"Running stage: {stage.name}")
        stage_result = {
            'name': stage.name,
            'status': 'running',
            'start_time': time.time(),
            'commands': [],
            'artifacts': [],
            'metrics': {}
        }
        try:
            if stage.parallel and len(stage.commands) > 1:
                # Run commands in parallel
                command_results = self.run_commands_parallel(stage.commands, stage.timeout)
            else:
                # Run commands sequentially
                command_results = []
                for command in stage.commands:
                    result = self.run_command(command, stage.timeout)
                    command_results.append(result)
                    # Stop on first failure if required
                    if stage.required and result['return_code'] != 0:
                        break
            stage_result['commands'] = command_results

            # Determine stage status
            failed_commands = [cmd for cmd in command_results if cmd['return_code'] != 0]
            if failed_commands and stage.required:
                stage_result['status'] = 'failed'
            elif failed_commands:
                stage_result['status'] = 'warning'
            else:
                stage_result['status'] = 'passed'

            # Extract metrics based on stage type
            stage_result['metrics'] = self.extract_stage_metrics(stage.name, command_results)
        except Exception as e:
            self.logger.error(f"Stage {stage.name} failed with exception: {str(e)}")
            stage_result['status'] = 'error'
            stage_result['error'] = str(e)
        finally:
            stage_result['execution_time'] = time.time() - stage_result['start_time']
            self.logger.info(f"Stage {stage.name} completed with status: {stage_result['status']}")
        return stage_result

    def run_command(self, command: str, timeout: int) -> Dict:
        """Run a single command with timeout"""
        self.logger.info(f"Executing: {command}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            return {
                'command': command,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': 0  # Would need to measure this separately
            }
        except subprocess.TimeoutExpired:
            self.logger.error(f"Command timed out after {timeout}s: {command}")
            return {
                'command': command,
                'return_code': -1,
                'stdout': '',
                'stderr': f'Command timed out after {timeout} seconds',
                'execution_time': timeout
            }
        except Exception as e:
            self.logger.error(f"Command failed with exception: {str(e)}")
            return {
                'command': command,
                'return_code': -1,
                'stdout': '',
                'stderr': str(e),
                'execution_time': 0
            }

    def run_commands_parallel(self, commands: List[str], timeout: int) -> List[Dict]:
        """Run multiple commands in parallel"""
        with ThreadPoolExecutor(max_workers=min(len(commands), 4)) as executor:
            futures = [
                executor.submit(self.run_command, command, timeout)
                for command in commands
            ]
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=timeout + 10)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Parallel command execution failed: {str(e)}")
                    results.append({
                        'command': 'unknown',
                        'return_code': -1,
                        'stdout': '',
                        'stderr': str(e),
                        'execution_time': 0
                    })
            return results

    def extract_stage_metrics(self, stage_name: str, command_results: List[Dict]) -> Dict:
        """Extract metrics from stage execution results"""
        metrics = {}
        for result in command_results:
            command = result['command']
            stdout = result['stdout']
            stderr = result['stderr']
            # Code coverage metrics
            if 'coverage report' in command:
                metrics.update(self.parse_coverage_output(stdout))
            # Pylint metrics
            elif 'pylint' in command:
                metrics.update(self.parse_pylint_output(stdout, stderr))
            # Flake8 metrics
            elif 'flake8' in command:
                metrics.update(self.parse_flake8_output(stdout, stderr))
            # Test metrics
            elif 'pytest' in command:
                metrics.update(self.parse_pytest_output(stdout, stderr))
            # Security metrics
            elif 'bandit' in command or 'safety' in command:
                metrics.update(self.parse_security_output(stdout, stderr))
        return metrics

    def parse_coverage_output(self, output: str) -> Dict:
        """Parse coverage report output"""
        metrics = {'coverage_percentage': 0}
        lines = output.split('\n')
        for line in lines:
            if 'TOTAL' in line and '%' in line:
                # Extract percentage from line like "TOTAL 100 25 75%"
                parts = line.split()
                for part in parts:
                    if '%' in part:
                        try:
                            metrics['coverage_percentage'] = float(part.replace('%', ''))
                            break
                        except ValueError:
                            pass
        return metrics

    def parse_pylint_output(self, stdout: str, stderr: str) -> Dict:
        """Parse pylint output for quality metrics"""
        metrics = {'pylint_score': 0, 'pylint_issues': 0}
        # Look for pylint score in output
        lines = (stdout + stderr).split('\n')
        for line in lines:
            if 'Your code has been rated at' in line:
                # Extract score from line like "Your code has been rated at 8.50/10"
                try:
                    score_part = line.split('at ')[1].split('/')[0]
                    metrics['pylint_score'] = float(score_part)
                except (IndexError, ValueError):
                    pass
        # Try to parse JSON output if available
        try:
            if stdout.strip().startswith('['):
                issues = json.loads(stdout)
                metrics['pylint_issues'] = len(issues)
        except json.JSONDecodeError:
            pass
        return metrics

    def parse_flake8_output(self, stdout: str, stderr: str) -> Dict:
        """Parse flake8 output"""
        metrics = {'flake8_issues': 0}
        lines = stdout.split('\n')
        issues = [line for line in lines if line.strip() and ':' in line]
        metrics['flake8_issues'] = len(issues)
        return metrics

    def parse_pytest_output(self, stdout: str, stderr: str) -> Dict:
        """Parse pytest output"""
        metrics = {
            'tests_total': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'tests_skipped': 0
        }
        output = stdout + stderr
        lines = output.split('\n')
        for line in lines:
            # Look for summary lines like "5 passed, 2 failed, 1 skipped"
            if 'passed' in line or 'failed' in line or 'skipped' in line:
                if 'passed' in line:
                    try:
                        passed = int(line.split('passed')[0].strip().split()[-1])
                        metrics['tests_passed'] = passed
                    except (ValueError, IndexError):
                        pass
                if 'failed' in line:
                    try:
                        failed = int(line.split('failed')[0].strip().split()[-1])
                        metrics['tests_failed'] = failed
                    except (ValueError, IndexError):
                        pass
                if 'skipped' in line:
                    try:
                        skipped = int(line.split('skipped')[0].strip().split()[-1])
                        metrics['tests_skipped'] = skipped
                    except (ValueError, IndexError):
                        pass
        metrics['tests_total'] = metrics['tests_passed'] + metrics['tests_failed'] + metrics['tests_skipped']
        return metrics

    def parse_security_output(self, stdout: str, stderr: str) -> Dict:
        """Parse security tool output"""
        metrics = {'security_issues': 0, 'security_high_issues': 0}
        try:
            if stdout.strip().startswith('{') or stdout.strip().startswith('['):
                data = json.loads(stdout)
                if isinstance(data, dict) and 'results' in data:
                    # Bandit format
                    metrics['security_issues'] = len(data['results'])
                    high_issues = [r for r in data['results'] if r.get('issue_severity') == 'HIGH']
                    metrics['security_high_issues'] = len(high_issues)
                elif isinstance(data, list):
                    # Safety format
                    metrics['security_issues'] = len(data)
        except json.JSONDecodeError:
            # Parse text output if JSON parse fails
            lines = stdout.split('\n')
            issue_lines = [line for line in lines if 'Issue:' in line or 'CRITICAL:' in line or 'HIGH:' in line]
            metrics['security_issues'] = len(issue_lines)
        return metrics

    def evaluate_quality_gates(self) -> Dict:
        """Evaluate all quality gates"""
        gate_results = {}
        # Collect all metrics from stages
        all_metrics = {}
        for stage_result in self.pipeline_results.get('stages', {}).values():
            all_metrics.update(stage_result.get('metrics', {}))
        # Evaluate each gate
        for gate_name, gate_config in self.config['quality_gates'].items():
            if not gate_config['enabled']:
                continue
            threshold = gate_config['threshold']
            current_value = self.get_metric_for_gate(gate_name, all_metrics)
            # Determine if gate passes
            if gate_name in ['code_coverage', 'test_success_rate', 'performance_score', 'security_score']:
                # Higher is better
                status = 'passed' if current_value >= threshold else 'failed'
            else:
                # Lower is better (e.g., code_quality_score for pylint)
                status = 'passed' if current_value >= threshold else 'failed'
            gate_results[gate_name] = {
                'threshold': threshold,
                'current_value': current_value,
                'status': status,
                'details': self.get_gate_details(gate_name, all_metrics)
            }
        return gate_results

    def get_metric_for_gate(self, gate_name: str, metrics: Dict) -> float:
        """Get the appropriate metric value for a quality gate"""
        if gate_name == 'code_coverage':
            return metrics.get('coverage_percentage', 0)
        elif gate_name == 'test_success_rate':
            total = metrics.get('tests_total', 0)
            passed = metrics.get('tests_passed', 0)
            return (passed / total * 100) if total > 0 else 0
        elif gate_name == 'performance_score':
            # This would be calculated based on performance test results
            return 85.0  # Placeholder
        elif gate_name == 'security_score':
            issues = metrics.get('security_issues', 0)
            high_issues = metrics.get('security_high_issues', 0)
            # Simple scoring: 100 - (issues * 5) - (high_issues * 10)
            score = 100 - (issues * 5) - (high_issues * 10)
            return max(0, score)
        elif gate_name == 'code_quality_score':
            return metrics.get('pylint_score', 0)
        return 0

    def get_gate_details(self, gate_name: str, metrics: Dict) -> Dict:
        """Get detailed information for a quality gate"""
        details = {}
        if gate_name == 'code_coverage':
            details = {
                'coverage_percentage': metrics.get('coverage_percentage', 0),
                'recommendation': 'Add more unit tests to increase coverage'
            }
        elif gate_name == 'test_success_rate':
            details = {
                'total_tests': metrics.get('tests_total', 0),
                'passed_tests': metrics.get('tests_passed', 0),
                'failed_tests': metrics.get('tests_failed', 0),
                'skipped_tests': metrics.get('tests_skipped', 0)
            }
        elif gate_name == 'security_score':
            details = {
                'total_issues': metrics.get('security_issues', 0),
                'high_severity_issues': metrics.get('security_high_issues', 0),
                'recommendation': 'Review and fix security vulnerabilities'
            }
        elif gate_name == 'code_quality_score':
            details = {
                'pylint_score': metrics.get('pylint_score', 0),
                'pylint_issues': metrics.get('pylint_issues', 0),
                'flake8_issues': metrics.get('flake8_issues', 0)
            }
        return details

    def generate_artifacts(self, pipeline_result: Dict):
        """Generate pipeline artifacts"""
        artifacts_dir = Path(self.config['artifacts']['output_directory'])
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        # Generate JSON report
        report_file = artifacts_dir / f'pipeline_report_{int(time.time())}.json'
        with open(report_file, 'w') as f:
            json.dump(pipeline_result, f, indent=2)
        pipeline_result['artifacts'].append(str(report_file))
        # Generate HTML report
        html_report = self.generate_html_report(pipeline_result)
        html_file = artifacts_dir / f'pipeline_report_{int(time.time())}.html'
        with open(html_file, 'w') as f:
            f.write(html_report)
        pipeline_result['artifacts'].append(str(html_file))
        # Generate quality gates summary
        gates_file = artifacts_dir / f'quality_gates_{int(time.time())}.json'
        with open(gates_file, 'w') as f:
            json.dump(pipeline_result.get('quality_gates', {}), f, indent=2)
        pipeline_result['artifacts'].append(str(gates_file))
        self.logger.info(f"Generated {len(pipeline_result['artifacts'])} artifacts")

    def generate_html_report(self, pipeline_result: Dict) -> str:
        """Generate HTML report"""
        status_colors = {
            'passed': '#28a745',
            'failed': '#dc3545',
            'warning': '#ffc107',
            'error': '#6f42c1',
            'pending': '#6c757d'
        }
        status_color = status_colors.get(pipeline_result['overall_status'], '#6c757d')
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Quality Pipeline Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: {status_color}; color: white; padding: 20px; border-radius: 5px; }}
        .stage {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .passed {{ border-left: 4px solid #28a745; }}
        .failed {{ border-left: 4px solid #dc3545; }}
        .warning {{ border-left: 4px solid #ffc107; }}
        .error {{ border-left: 4px solid #6f42c1; }}
        .gate {{ margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 3px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .metric {{ text-align: center; padding: 10px; background: #e9ecef; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Quality Pipeline Report</h1>
        <p>Status: <strong>{pipeline_result['overall_status'].upper()}</strong></p>
        <p>Commit: {pipeline_result['commit'][:8]}</p>
        <p>Branch: {pipeline_result['branch']}</p>
        <p>Execution Time: {pipeline_result['execution_time']:.2f}s</p>
    </div>

    <h2>Quality Gates</h2>
"""
        for gate_name, gate_result in pipeline_result.get('quality_gates', {}).items():
            gate_status = gate_result['status']
            gate_class = gate_status
            html += f"""
    <div class="gate {gate_class}">
        <h3>{gate_name.replace('_', ' ').title()}</h3>
        <p>Status: <strong>{gate_status.upper()}</strong></p>
        <p>Threshold: {gate_result['threshold']}</p>
        <p>Current Value: {gate_result['current_value']}</p>
    </div>
"""
        html += "<h2>Stages</h2>"
        for stage_name, stage_result in pipeline_result.get('stages', {}).items():
            stage_status = stage_result['status']
            stage_class = stage_status
            html += f"""
    <div class="stage {stage_class}">
        <h3>{stage_name}</h3>
        <p>Status: <strong>{stage_status.upper()}</strong></p>
        <p>Execution Time: {stage_result['execution_time']:.2f}s</p>

        <h4>Metrics</h4>
        <div class="metrics">
"""
            for metric_name, metric_value in stage_result.get('metrics', {}).items():
                html += f"""
            <div class="metric">
                <strong>{metric_name.replace('_', ' ').title()}</strong><br>
                {metric_value}
            </div>
"""
            html += """
        </div>
    </div>
"""
        html += """
</body>
</html>
"""
        return html

    def send_notifications(self, pipeline_result: Dict):
        """Send pipeline notifications"""
        if pipeline_result['overall_status'] in ['failed', 'error']:
            self.send_slack_notification(pipeline_result)
            self.send_email_notification(pipeline_result)

    def send_slack_notification(self, pipeline_result: Dict):
        """Send Slack notification"""
        webhook_url = self.config['notifications'].get('slack_webhook')
        if not webhook_url:
            return
        status = pipeline_result['overall_status']
        color = '#dc3545' if status in ['failed', 'error'] else '#28a745'
        message = {
            "attachments": [
                {
                    "color": color,
                    "title": "Quality Pipeline Report",
                    "fields": [
                        {
                            "title": "Status",
                            "value": status.upper(),
                            "short": True
                        },
                        {
                            "title": "Branch",
                            "value": pipeline_result['branch'],
                            "short": True
                        },
                        {
                            "title": "Commit",
                            "value": pipeline_result['commit'][:8],
                            "short": True
                        },
                        {
                            "title": "Execution Time",
                            "value": f"{pipeline_result['execution_time']:.2f}s",
                            "short": True
                        }
                    ]
                }
            ]
        }
        try:
            import requests
            requests.post(webhook_url, json=message)
            self.logger.info("Slack notification sent")
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {str(e)}")

    def send_email_notification(self, pipeline_result: Dict):
        """Send email notification"""
        recipients = self.config['notifications'].get('email_recipients', [])
        if not recipients:
            return
        # Email implementation would go here
        self.logger.info(f"Email notification would be sent to {len(recipients)} recipients")

    def cleanup_old_artifacts(self):
        """Cleanup old artifacts based on retention policy"""
        retention_days = self.config['artifacts']['retention_days']
        artifacts_dir = Path(self.config['artifacts']['output_directory'])
        if not artifacts_dir.exists():
            return
        cutoff_time = time.time() - (retention_days * 24 * 60 * 60)
        cleaned_count = 0
        for file_path in artifacts_dir.glob('*'):
            if file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    cleaned_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to cleanup {file_path}: {str(e)}")
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old artifacts")


# CLI interface
def main():
    """Main CLI interface for quality pipeline"""
    import argparse
    parser = argparse.ArgumentParser(description='Quality Assurance Pipeline')
    parser.add_argument('--config', default='qa/pipeline_config.yaml', help='Pipeline configuration file')
    parser.add_argument('--stage', help='Run specific stage only')
    parser.add_argument('--cleanup', action='store_true', help='Cleanup old artifacts')
    parser.add_argument('--report-only', action='store_true', help='Generate report from existing results')
    args = parser.parse_args()
    pipeline = QualityPipeline(args.config)
    if args.cleanup:
        pipeline.cleanup_old_artifacts()
        return
    if args.report_only:
        # Generate report from existing results (not implemented)
        print("Report generation from existing results not implemented yet")
        return
    # Run full pipeline or specific stage
    if args.stage:
        # Run specific stage only (not implemented)
        print(f"Running specific stage not implemented yet: {args.stage}")
    else:
        results = pipeline.run_pipeline()
        print(f"\nPipeline completed with status: {results['overall_status']}")
        print(f"Execution time: {results['execution_time']:.2f} seconds")
        print(f"Artifacts generated: {len(results.get('artifacts', []))}")
        # Print quality gate summary
        print("\nQuality Gates:")
        for gate_name, gate_result in results.get('quality_gates', {}).items():
            status_emoji = "✅" if gate_result['status'] == 'passed' else "❌"
            print(f"  {status_emoji} {gate_name}: {gate_result['current_value']:.1f} (threshold: {gate_result['threshold']})")

if __name__ == "__main__":
    main()
