import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import torch


class BertModel:
#Initalize method. Created with the Bert large model, output_hidden_states=True, and the BertTokenizer
    def __init__(self, model=BertModel.from_pretrained('bert-large-uncased',
                                                       output_hidden_states=True,
                                                       # Whether the model returns all hidden-states.
                                                       ), evaluation=False,
                 tokenizer=BertTokenizer.from_pretrained('bert-large-uncased')):

        self.model = model
        self.evaluation = evaluation
        self.tokenizer = tokenizer
        

#Method to remove evaluation mode on the model
    def configure_evaluation(self, evaluation):
        if evaluation:
            return False
        else:
            return True

#Creating the size of the finale layer. 1024 for large and 756 for base
    def shape_embeddings(self,sentence_embeddings):
        sentence_embeddings = np.array([np.array(x) for x in sentence_embeddings])
        sentence_embeddings = sentence_embeddings.reshape(-1, 1024)
        return sentence_embeddings


    def transform(self, sentences):
        sentence_embeddings = []
        for sentence in sentences:
            # Adding tags to sentences
            marked_text = "[CLS] " + sentence + " [SEP]"
            tokenized_text = self.tokenizer.tokenize(marked_text)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

            # Segment sentence
            segments_ids = [1] * len(tokenized_text)

            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            if not self.evaluation:
                # Put the model in "evaluation" mode, meaning feed-forward operation.
                self.model.eval()
                evaluation = self.configure_evaluation(self.evaluation)
            with torch.no_grad():
                outputs = self.model(tokens_tensor, segments_tensors)
                # Evaluating the model will return a different number of objects based on
                # how it's  configured in the `from_pretrained` call earlier. In this case,
                # becase we set `output_hidden_states = True`, the third item will be the
                # hidden states from all layers.
                hidden_states = outputs[2]
            token_vecs = hidden_states[-2][0]
            # Calculate the average of all 22 token vectors.
            sentence_embedding = torch.mean(token_vecs, dim=0)
            sentence_embeddings.append(sentence_embedding)
            #Calling shap embeddings
        sentence_embeddings = self.shape_embeddings(sentence_embeddings)
        return sentence_embeddings

    def get_embeddings(self, sentences):
        sentence_embeddings = self.transform(sentences)
        return sentence_embeddings
