# nlg-a1
The first attempt about nature language generation; use simple LSTM RNN to generate text in both character level and word level\n
files starting with 'train' are used to train the constructed neural network.\n
files starting with 'demo' are used to test the trained neural network.\n
files with 'char' are designed for character-based NLG.\n
files with 'word' are desined for word-based NLG.\n
'create_vocab' is used to create a universal vocabulary list which can be directly used in training process.\n
'n-gram_LM' is the n-gram language module which is used to calculate the perplexity.\n
'txtprocessing' is used to preprocess the texts.\n
There are some special neural network designs for attempts:\n
'demo_char_addsigns' uses two different character-based RNN. One only contains the letters in the alphabet. The other only contains some commom signs. In this program, it will firstly generate letters, and then, add signs to devide these chracters. At present, it seems it does not work well.\n
'train_charCN' and 'train_vec_charCN' are attempts to process chinese characters. Just for try:)\n
'char_lower' means all the characters are lowercase.\n
Some training data I used is in the data doc.\n
