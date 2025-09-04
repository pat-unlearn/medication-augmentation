# 🚀 Medication Augmentation CLI - Quick Reference

## **The Best Way to Use the CLI**

Activate the virtual environment once, then use clean commands:

```bash
# 1. Activate the virtual environment (one-time per session)
source .venv/bin/activate

# 2. Now use simple, clean commands
med-aug info
med-aug diseases list
med-aug pipeline run data/sample_nsclc.csv --disease nsclc --output ./results
```

---

## **Quick Demo Commands**

### **Setup (First Time)**
```bash
# Navigate to project directory
cd /Users/pat/Developer/medication-augmentation

# Activate virtual environment
source .venv/bin/activate

# You're ready! Now use the cli
```

### **1. Test & System Info**
```bash
# Test CLI is working
med-aug test

# Show system information
med-aug info

# Show help
med-aug --help
```

### **2. Disease Management**
```bash
# List available disease modules
med-aug diseases list

# Show NSCLC module details
med-aug diseases show nsclc

# Validate module configuration
med-aug diseases validate nsclc
```

### **3. Data Analysis**
```bash
# Find medication columns in dataset
med-aug pipeline analyze data/sample_nsclc.csv

# Extract medications from specific column
med-aug pipeline extract data/sample_nsclc.csv AGENT --output results.json
```

### **4. Pipeline Execution**
```bash
# Basic pipeline run (with LLM by default) - augments existing conmeds file
med-aug pipeline run data/sample_nsclc.csv \
  --conmeds data/conmeds_defaults.yml \
  --disease nsclc \
  --output ./results

# Advanced run with evaluation
med-aug pipeline run data/sample_nsclc.csv \
  --conmeds data/conmeds_defaults.yml \
  --disease nsclc \
  --evaluate \
  --output ./results

# Disable LLM if needed (not recommended)
med-aug pipeline run data/sample_nsclc.csv \
  --conmeds data/conmeds_defaults.yml \
  --disease nsclc \
  --no-llm \
  --output ./results
```

---

## **Alternative: One-Off Commands**

If you don't want to activate the environment:

```bash
# Use uv run for one-off commands
uv run med-aug info
uv run med-aug diseases list
```
