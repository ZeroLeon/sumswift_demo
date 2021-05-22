import torch
import streamlit as st
import requests
import os


from transformers import MT5ForConditionalGeneration
import jieba
from transformers import BertTokenizer
# from torch._six import container_abcs, string_classes, int_classes
import torch
# from torch.utils.data import DataLoader, Dataset
# import re



class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


tokenizer = T5PegasusTokenizer.from_pretrained('imxly/t5-pegasus-small')
max_len =512
device = 'cpu'



def load_data(filename):
    """加载数据
    单条格式：(标题, 正文)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            title, content = l.strip().split('\t')
            D.append((title, content))
    return D



def generate(text,model, max_length=45):
    max_content_length = max_len - max_length
    feature = tokenizer.encode(text, return_token_type_ids=True, return_tensors='pt',
                               max_length=512)
    feature = {'input_ids': feature}
    feature = {k: v.to(device) for k, v in list(feature.items())}

    gen = model.generate(max_length=max_length, eos_token_id=tokenizer.sep_token_id,
                         decoder_start_token_id=tokenizer.cls_token_id,
                         **feature).cpu().numpy()[0]
    gen = gen[1:]
    gen = tokenizer.decode(gen, skip_special_tokens=True).replace(' ', '')
    return gen



@st.cache(allow_output_mutation=True)
def get_learner(file_name='summary_model.pt'):
    model = torch.load('summary_model.pt',map_location='cpu')
    model.to(device)
    model.eval()
    return model


def download_files(URL):
    with open("summary_model.pt", "wb") as model:
        r = requests.get(URL)
        model.write(r.content)
    try:
        assert(os.path.getsize("summary_model.pt") > 15000)
        st.success('Model successfully downloaded!')
    except:
        st.warning("Something wrong happened")

# https://www.dropbox.com/s/beh1cywvuq4urln/summary_model.pt?dl=1

def write():
    st.title('Model for Image Classification')
    if (not os.path.isfile('summary_model.pt') or os.path.getsize("summary_model.pt") < 15000):
        ph = st.empty()
        ph2 = st.empty()
        ph3 = st.empty()
        ph.warning('Please download the model file')
        URL = ph3.text_input('Please input the direct download link of your model file.','')
        if ph2.button('Download'):
            ph.empty()
            ph3.empty()
            ph2.text('Downloading...')
            try:
                download_files(URL)
                ph2.text('Download completed')
                st.button("Next Stage")
            except Exception as e:
                st.error('Not a correct URL!')
                print(str(e))

    else:
        st.success("Model already downloaded")

        txt_data = st.text_area('Please input text for testing')
        if txt_data == None:
                st.warning('Please input data...')
        else:
            # st.text_area(txt_data,use_column_width=True)
            check = st.button('generate')
            if check:
                file_name = 'summary_model.pt'
                model  = get_learner(file_name)
                result = generate(txt_data,model,max_length=45)
                # img = PILImage.create(img_data)
                # result = learn.predict(txt_data)
                st.write(result)


if __name__ == "__main__":
    write()