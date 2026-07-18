@echo off
echo Setting up CrowdWisdomTrading Quantitative ML Pipeline...

echo Installing Python dependencies...
python -m pip install -r requirements.txt

echo Installing Playwright browsers...
playwright install chromium

echo Setup complete!
echo You can now run the pipeline with: python main.py pipeline
pause
