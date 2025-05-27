
path_model=r'/data/@Zilong_Works/Data/SourceData/Model_Weight_Cache/Seg/GLaMM'

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model=path_model)
pass