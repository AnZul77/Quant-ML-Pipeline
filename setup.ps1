Write-Output "Setting up CrowdWisdomTrading Quantitative ML Pipeline..."

Write-Output "Installing Python dependencies..."
python -m pip install -r requirements.txt

Write-Output "Installing Playwright browsers..."
playwright install chromium

Write-Output "Setup complete!"
Write-Output "You can now run the pipeline with: python main.py pipeline"
