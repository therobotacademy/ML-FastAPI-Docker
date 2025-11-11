## Train the model (on the host)

Run `train_model.py` for generating the model prediction file, only needed if `./src/models/iris_model.joblib` does not exist

This script generates the output file `iris_model.joblib` that will be to make predictions:

```python
python train_model.py
```

## Building and Running with Docker

1. **Build** the Docker image:
   ```bash
   docker build -t ml-iris-repl .
   ```
2. **Run** a container from that image:
   ```bash
   docker run --rm -it --name ml-iris-repl-app ml-iris-repl
   ```
3. An interactive terminal opens so that you can make predictions

## Execution Modes

First run the container as a bash terminal:

```bash
   docker run --rm --name ml-iris-repl-app ml-iris-repl bash
```

Then use any of the options below.

### 1. Local Python (Interactive Terminal Mode)

Run predictions directly in the terminal — no HTTP server needed.

```bash
python main.py
```

* Loads `model.joblib`
* Prompts the user for 4 comma-separated Iris features
  *(sepal_length, sepal_width, petal_length, petal_width)*
* Prints predicted species and class probabilities.

**Example session:**

```

✅ Loaded model: model.joblib
💬 Interactive Iris prediction
Features> 5.1,3.5,1.4,0.2
🔮 Prediction: setosa (proba → setosa=0.98, versicolor=0.02, virginica=0.00)
```

**Purpose:**
Quick manual testing or classroom demo without running FastAPI.

### 2. One-Shot Prediction Mode

Pass features directly as command-line arguments.

```bash
python main.py --features 6.1 2.8 4.7 1.2
```

**Output:**

```
✅ Loaded model: /src/models/iris_model.joblib
🔮 Prediction: versicolor  (proba → versicolor=0.990, virginica=0.010, setosa=0.000)
```

**Purpose:**
Automate single predictions in scripts or shell pipelines.

### 3. Batch Prediction from CSV

Predict labels for multiple samples in a `.csv` file (no header, 4 columns per row).

```bash
python main.py --csv data/samples.csv
```

**Output:**

```
✅ Loaded model: /src/models/iris_model.joblib
row 1: setosa
row 2: setosa
row 3: virginica
...
```

**Purpose:**
Batch testing or validation of multiple feature rows.

### **Help & Examples**

Display examples or metadata:

```bash
python main.py --example
```

### Summary Table

| Mode                   | Command                                       | Description                          |
| ---------------------- | --------------------------------------------- | ------------------------------------ |
| **Interactive**  | `python main.py`                            | Manual prediction via terminal input |
| **One-Shot**     | `python main.py --features 5.1 3.5 1.4 0.2` | Predict from CLI arguments           |
| **Batch (CSV)**  | `python main.py --csv samples.csv`          | Predict multiple rows                |
| **Example/Help** | `python main.py --example`                  | Show usage examples                  |

## ANNEX: Iris dataset description

The **Iris dataset** — which this `samples.csv` is based on — uses the following four feature columns:

| # | Column name      | Description                        | Example value |
| :-: | :--------------- | :--------------------------------- | :------------ |
| 1 | `sepal_length` | Length of the sepal in centimeters | 5.1           |
| 2 | `sepal_width`  | Width of the sepal in centimeters  | 3.5           |
| 3 | `petal_length` | Length of the petal in centimeters | 1.4           |
| 4 | `petal_width`  | Width of the petal in centimeters  | 0.2           |

When you enter data manually or via `--features`, always keep this order:

```
[sepal_length, sepal_width, petal_length, petal_width]
```

Example:

```bash
python main.py --features 6.1 2.8 4.7 1.2
```
