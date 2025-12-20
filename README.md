# NDNF Pipeline

A shared DataJoint pipeline for the Neuronal Diversity in Network Function (NDNF) lab. This package provides common schemas and utilities for managing lab data, including subjects, surgeries, mouse lines, and experimental metadata.

## Installation

### Prerequisites

- Python >= 3.13
- MySQL server (for DataJoint database)
- Graphviz (for schema visualization)

### Install from GitHub

```bash
pip install git+https://github.com/rozmar/NDNF-pipeline.git
```

### Install for Development

```bash
git clone https://github.com/rozmar/NDNF-pipeline.git
cd NDNF-pipeline
pip install -e .
```

### System Dependencies

For schema visualization, install Graphviz:

```bash
# macOS
brew install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz

# Windows
# Download from https://graphviz.org/download/
```

## Configuration

### DataJoint Configuration

Create a `dj_local_conf.json` file with your database credentials:

```json
{
    "database.host": "cs16-datajoint.koki.local",
    "database.user": "your-username",
    "database.password": "your-password",
    "database.use_tls": false,
    "loglevel": "INFO",
    "safemode": true,
    "display.limit": 7,
    "display.width": 14,
    "display.show_tuple_count": true,
    "project": "behavior",
    "custom": {
        "database.prefix": "pipeline_"
        },
    "metadata.spreadsheet_names":["NDNF viruses plasmids",
                                  "NDNF animals",
                                  "NDNF experimenters",
                                  "NDNF procedures"],
    "path.metadata": "path-to-metadata",
    "path.google_creds_json": "path-to-credentials-json"
}
```

Load the configuration in your code:

```python
import datajoint as dj
dj.config.load('path/to/dj_local_conf.json')
dj.conn()
```

### Google Sheets Integration (Optional)

For Google Sheets metadata synchronization, you'll need:

1. Google API credentials (JSON file)
2. Service account with appropriate permissions


## Usage

### Basic Import

```python
from ndnf_pipeline import lab
from ndnf_pipeline.utils import update_metadata
```

### Visualize Schema

```python
import datajoint as dj
from ndnf_pipeline import lab

# Display entire schema
dj.Diagram(lab.schema)
```

### Sync Metadata from Google Sheets

```python
from ndnf_pipeline.utils import update_metadata

update_metadata(
    notebook_name='Lab Metadata',
    metadata_dir='/path/to/metadata',
    google_creds_json='/path/to/google_creds.json'
)
```

## Acknowledgments

Built with [DataJoint](https://datajoint.com/) for Python.