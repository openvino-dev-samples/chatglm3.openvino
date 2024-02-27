# chatglm3.openvino

**1. Set-up the environments:**

```
python3 -m venv openvino_env

source openvino_env/bin/activate

python3 -m pip install --upgrade pip

pip install wheel setuptools

pip install -r requirements.txt
```

**2. Convert model:**

```
python3 convert.py
```

**3. Quantize model:**

```
python3 quantize.py
```

**4. Run generation:**

```
python3 generate.py -m "./chatglm3_compressed"
```