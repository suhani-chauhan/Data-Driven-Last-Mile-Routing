# Runs the interactive dashboard out of the box using the small, git-committed
# data/deploy/ subset (same one Streamlit Community Cloud uses) -- no need to
# download the full ~568MB data/processed/ just to see the container run.
#
# For the CLI pipeline (data_pipeline.py, model_apply.py, model_apply_fleet.py,
# model_score.py), which need the full data/processed/, mount your local data/
# directory and override the command -- see README.md's Docker section for
# the exact syntax per shell (macOS/Linux, Windows Git Bash, PowerShell).
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/deploy/ data/deploy/

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
