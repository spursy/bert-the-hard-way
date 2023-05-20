## BERT-THE-HARD-WAY


```bash
// python package
pip install -r requirements.txt

pip freeze > requirements.txt

// verify python transformers package
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('you are so handsome!'))"

// install transformers related tools
brew reinstall pkg-config
brew install cmake
```