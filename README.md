# nlg-a1
The first attempt about nature language generation; use simple LSTM RNN to generate text in both character level and word level\\
files starting with 'train' are used to train the constructed neural network.\\
files starting with 'demo' are used to test the trained neural network.\\
files with 'char' are designed for character-based NLG.\\
files with 'word' are desined for word-based NLG.\\
'create_vocab' is used to create a universal vocabulary list which can be directly used in training process.\\
'n-gram_LM' is the n-gram language module which is used to calculate the perplexity.\\
'txtprocessing' is used to preprocess the texts.\\
There are some special neural network designs for attempts:
'demo_char_addsigns' uses two different character-based RNN. One only contains the letters in the alphabet. The other only contains some commom signs. In this program, it will firstly generate letters, and then, add signs to devide these chracters. At present, it seems it does not work well.
'train_charCN' and 'train_vec_charCN' are attempts to process chinese characters. Just for try:)
'char_lower' means all the characters are lowercase.
Some training data I used is in the data doc.
