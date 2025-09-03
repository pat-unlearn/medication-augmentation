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

**✨ Clean Output:** Professional formatting with no messy log messages!

---

## **Quick Demo Commands**

### **Setup (First Time)**
```bash
# Navigate to project directory
cd /Users/pat/Developer/medication-augmentation

# Activate virtual environment  
source .venv/bin/activate

# You're ready! Now use clean commands:
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

## **Why This Approach is Better**

1. **Cleanest Commands**: Just `med-aug info` instead of `uv run med-aug info`
2. **Standard Practice**: Activating virtual environments is the Python standard
3. **Professional Look**: Clean, simple commands perfect for demos and documentation
4. **Team Friendly**: Easy for colleagues to remember and use

---

## **Alternative: One-Off Commands**

If you don't want to activate the environment:

```bash
# Use uv run for one-off commands
uv run med-aug info
uv run med-aug diseases list
```

But activating the environment once is cleaner for multiple commands.

---

## **Setup Instructions**

### **Initial Setup (One Time)**
```bash
# 1. Navigate to project directory
cd /Users/pat/Developer/medication-augmentation

# 2. Install dependencies and CLI
uv install
uv pip install -e .

# 3. Test installation
source .venv/bin/activate
med-aug info
```

### **Daily Usage**
```bash
# Activate environment
source .venv/bin/activate

# Use clean commands
med-aug info
med-aug diseases list
# ... any other commands
```

---

## **5-Minute Demo Script for Your Team**

### **Option 1: Real Clinical Data (Recommended)**
```bash
# Setup
cd /Users/pat/Developer/medication-augmentation
source .venv/bin/activate

# Show capabilities
med-aug info
med-aug diseases list

# Analyze REAL clinical trial data (20K+ rows, 2K+ medications)
med-aug pipeline analyze data/rpmed.csv

# Show what it found
med-aug pipeline extract data/rpmed.csv DRUGDTXT --output ./real_meds.json
echo "Found these top medications in real clinical data:"
cat real_meds.json | head -20

# Run complete augmentation on real data
med-aug pipeline run data/rpmed.csv \
  --conmeds data/conmeds_defaults.yml \
  --disease nsclc \
  --output ./demo_real

# Show results
ls -la demo_real/
head -10 demo_real/conmeds_augmented.yml
```

### **Option 2: Timeline Treatment Data**
```bash
# Analyze longitudinal treatment data
med-aug pipeline analyze data/timeline_treatments.csv

# Extract cancer treatments 
med-aug pipeline extract data/timeline_treatments.csv AGENT --output ./timeline_meds.json

# Run augmentation
med-aug pipeline run data/timeline_treatments.csv \
  --conmeds data/conmeds_defaults.yml \
  --disease nsclc \
  --output ./demo_timeline
```

### **Option 3: Synthetic Demo Data (Safe for Presentations)**
```bash
# Use synthetic data if real data has privacy concerns
med-aug pipeline analyze data/sample_nsclc.csv
med-aug pipeline run data/sample_nsclc.csv \
  --conmeds data/conmeds_defaults.yml \
  --disease nsclc \
  --output ./demo_synthetic
```

**Why Real Data is Better:**
- ✅ **Authentic**: 20,384 real clinical records with 2,290 unique medications
- ✅ **Impressive Scale**: Shows system handles large-scale clinical data  
- ✅ **Real Challenges**: Contains actual messy medication names from trials
- ✅ **Credible Demo**: Your clinical team will recognize authentic scenarios

**Output:** Production-ready augmented conmeds.yml file from real clinical data  
**Impression:** Robust system that handles authentic clinical datasets