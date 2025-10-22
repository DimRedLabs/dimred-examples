# DimRed Examples

A repository of examples and experiments demonstrating the DimRed API for prompt tuning and optimization.

## Overview

This repository contains:
- **client.py** - Python client for interacting with the DimRed API
- **examples/** - Example scripts and notebooks demonstrating the API workflow
- **data/** - Sample datasets for testing

## Getting Started

### Prerequisites

- Python 3.8 or higher
- A DimRed API key (get one at [dimred.com](https://dimred.com))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DimRedLabs/dimred-examples.git
   cd dimred-examples
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**

   Create a `.env` file in the project root:
   ```bash
   echo "DIMRED_API_KEY=your_api_key_here" > .env
   ```

   Replace `your_api_key_here` with your actual DimRed API key.

### Running the Examples

#### Python Script

Run the complete tuning workflow from the command line:

```bash
python examples/run_tuning.py
```

The script will automatically load your API key from the `.env` file. You can also pass it directly:

```bash
python examples/run_tuning.py --api-key YOUR_API_KEY
```

Additional options:
```bash
python examples/run_tuning.py --help
```

#### Jupyter Notebook

Launch the interactive notebook:

```bash
jupyter notebook examples/run_tuning.ipynb
```

The notebook demonstrates the same workflow with step-by-step explanations.

## What the Examples Do

The example workflow demonstrates:

1. **Create a project** - Set up a new DimRed project
2. **Create a dataset** - Initialize a dataset for your task
3. **Add datapoints** - Upload training examples with inputs and expected outputs
4. **Create a prompt** - Define the initial system prompt
5. **Create a metric** - Define how to evaluate prompt performance
6. **Run tuning** - Start the automated prompt optimization process
7. **Wait for completion** - Poll until tuning finishes
8. **Get results** - Retrieve the best performing prompt

The included example focuses on financial crime detection, classifying whether individuals mentioned in news articles are perpetrators of financial crimes.

## Project Structure

```
dimred-examples/
├── client.py              # DimRed API client library
├── examples/
│   ├── run_tuning.py      # Command-line example script
│   └── run_tuning.ipynb   # Jupyter notebook example
├── data/
│   └── example.json       # Sample dataset
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## API Client Usage

You can import and use the client in your own scripts:

```python
from client import DimRedAPIClient

# Initialize the client
client = DimRedAPIClient(api_key="your_api_key")

# Create a project
project_id = client.create_project(
    project_name="My Project",
    project_description="Description here"
)

# Continue with your workflow...
```

## Environment Variables

- `DIMRED_API_KEY` - Your DimRed API key (required)

## Security Notes

- Never commit your `.env` file or expose your API key
- The `.gitignore` file is configured to exclude `.env` files
- Keep your API key secure and rotate it if compromised

## Support

For questions or issues:
- Visit [dimred.com](https://dimred.com)
- Check the [documentation](https://docs.dimred.com)
- Open an issue in this repository

## License

This project is provided as-is for demonstration purposes.
