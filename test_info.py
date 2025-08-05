
import torch
from parlai.core.opt import Opt


path_finetuned = "models/normal_transformer_finetuned.mdl"
path_pretrained = "models/normal_transformer_pretrained.mdl"

m_finetuned = torch.load(path_finetuned, map_location='cpu')
m_pretrained = torch.load(path_pretrained, map_location='cpu')


def print_info(model_data):

    opt = Opt(vars(model_data['opt']))

    print("model:", opt['model'])
    print("model_file:", opt.get('model_file', None))
    print("dict_file:", opt.get('dict_file', None))
    print("model-type:", opt.get('model', None))
    print("n_layers:", opt.get('n_layers', None))
    print("transformer_dim:", opt.get('transformer_dim', None))
    print("transformer_n_heads:", opt.get('transformer_n_heads', None))
    print("transformer_dropout:", opt.get('transformer_dropout', None))
    print("hidden:", opt.get('hidden', None))
    print("num_epochs:", opt.get('num_epochs', None))

    print(68 * '*')
    print("dataset_name:", opt.get('dataset_name', None))
    print("reddit:", opt.get('reddit', None))
    print("max_sent_len:", opt.get('max_sent_len', None))
    print("max_hist_len:", opt.get('max_hist_len', None))
    print("dict_max_words:", opt.get('dict_max_words', None))
    print("encoder_type:", opt.get('encoder_type', None))
    #print("embedding type:", opt.get('embedding_type', None))
    print("embeddings_size:", opt.get('embeddings_size', None))
    print("embeddings:", opt.get('embeddings', None))

    # for k, v in opt.items():
    #     print(f"{k}: {v}")


if __name__ == "__main__":

    print_info(m_pretrained)
    print(68 * '*')
    print_info(m_finetuned)
