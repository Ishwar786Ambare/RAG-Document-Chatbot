import logging
try:
    import langchain
    print("SUCCESS: imported langchain")
    print(dir(langchain))
    import langchain.chains
    print("SUCCESS: imported langchain.chains")
except Exception as e:
    import traceback
    traceback.print_exc()
