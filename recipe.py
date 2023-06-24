from cook import create_task


create_task("requirements", action="pip-compile -v", targets=["requirements.txt"],
            dependencies=["requirements.in", "setup.py"])

create_task("tests", action="pytest -v --cov=summaries --cov-report=html --cov-fail-under=100")
