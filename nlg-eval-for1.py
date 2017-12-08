from nlgeval import compute_individual_metrics
import codecs

filename1 = "op.txt"
references = codecs.open(filename1,'r','utf-8').read()

filename2 = "op.txt"
hypothesis = codecs.open(filename2,'r','utf-8').read()
metrics_dict = compute_individual_metrics('||<|>||'.join(references), hypothesis)
