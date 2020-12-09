# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from gensim import models,corpora
from IPython.display import display,clear_output
from ipywidgets import IntSlider,Button,Output,ToggleButtons,HBox,VBox

class recommender():
    def __init__(self,path):
        df=pd.read_csv(path+'./data/behavior_dev.csv')
        self.data=df['content'].values
        self.sentences=self.split_data(self.data)

        self.n_topics=12
        self.n_data=len(self.data)
        self.n_sentences=len(self.sentences)

        self.lda=models.LdaModel.load(path+'./models/lda.model')

        self.model=nn.Sequential(nn.Linear(self.n_topics,128),
                                 nn.Sigmoid(),
                                 nn.Linear(128,128),
                                 nn.Sigmoid(),
                                 nn.Linear(128,self.n_data))
        
        self.model_s=nn.Sequential(nn.Linear(self.n_topics,128),
                                   nn.Sigmoid(),
                                   nn.Linear(128,128),
                                   nn.Sigmoid(),
                                   nn.Linear(128,self.n_sentences))
        
        self.model.load_state_dict(torch.load(path+'./models/mlp_model.pt',map_location=torch.device('cpu')))
        self.model_s.load_state_dict(torch.load(path+'./models/mlp_model_s.pt',map_location=torch.device('cpu')))

        self.chars=['자기주도적 학습 태도',
                    '직업과의 연계성 전망',
                    '주관 및 의사 표현',
                    '성실성',
                    '관계지향성',
                    '자기 관심 분야 선호',
                    '책임감',
                    '학업에 대한 목표의식과 노력',
                    '나눔과 배려',
                    '규칙준수',
                    '리더십',
                    '협동심']

        corpus=corpora.MmCorpus(path+'./data/corpus.mm')
        corpus_s=corpora.MmCorpus(path+'./data/corpus_s.mm')
        self.score=self.topic_score(corpus)
        self.score_s=self.topic_score(corpus_s)

    def topic_score(self,corpus):
        topic_scores=[]
        for i in range(len(corpus)):
            topic_scores.append(np.array(self.lda.get_document_topics(corpus[i],minimum_probability=0))[:,1])
        return np.array(topic_scores)
        

    def topn_words(self,n_words):
        topic_data=[]
        for i in range(self.n_topics):
            topic_data.append(np.array(self.lda.show_topic(i,n_words))[:,0])           
        index=[]
        for i in range(self.n_topics):
            index.append(self.chars[i])
        
        return pd.DataFrame(topic_data,index=index).transpose()

    def split_data(self,data):
        sentences=[]
        for text in data:
            for s in text.split('. '):
                if len(s)!=0:
                    if s[-1]!='.':
                      sentences.append(s+'.')
                    else:
                      sentences.append(s)
        return np.unique(sentences)

    def split_text(self,text):
        sentences=[]
        for s in text.split('. '):
            if len(s)!=0:
                if s[-1]!='.':
                  sentences.append(s+'.')
                else:
                  sentences.append(s)
        return np.array(sentences)

    def normalize(self,x):
        return x/np.sum(x)

    def predict(self,x,n):
        with torch.no_grad():
            x=torch.tensor(self.normalize(x),dtype=torch.float)
            y=self.model(x)
            probs=F.softmax(y,dim=-1).data.numpy()
        index=np.argsort(probs)[::-1][:n]

        print('\n')
        print('정확도 상위 %s개의 문서 검색 결과'%n+'\n')
        for i in range(n):
            score_index=np.argsort(self.score[index[i]])[::-1][:3]
            sentences=self.split_text(self.data[index[i]])

            print('%s,'%(i+1), '정확도: {:.2f}%'.format(probs[index[i]]*100))
            print('='*200)
            for s in sentences:
                print(s)
            print('-'*200)
            for j in range(3):
                print('%s: %.3f'%(self.chars[j],self.score[index[i]][score_index[j]]))
            print('='*200+'\n')

    def predict_s(self,x,n):
        with torch.no_grad():
            x=torch.tensor(self.normalize(x),dtype=torch.float)
            y=self.model_s(x)
            probs=F.softmax(y,dim=-1).data.numpy()
        index=np.argsort(probs)[::-1][:n]

        print('\n')
        print('정확도 상위 %s개의 문장 검색 결과'%n+'\n')
        for i in range(n):
            score_index=np.argsort(self.score_s[index[i]])[::-1][:3]
            
            print('%s,'%(i+1), '정확도: {:.2f}%'.format(probs[index[i]]*100))
            print('='*200)
            print(self.sentences[index[i]])
            print('-'*200)
            for j in range(3):
                print('%s: %.3f'%(self.chars[j],self.score_s[index[i]][score_index[j]]))
            print('='*200+'\n')

    def call(self):
        style={'description_width':'initial'}
        c1=IntSlider(value=3,min=1,max=5,step=1,description=self.chars[0],style=style)
        c2=IntSlider(value=3,min=1,max=5,step=1,description=self.chars[1],style=style)
        c3=IntSlider(value=3,min=1,max=5,step=1,description=self.chars[2],style=style)
        c4=IntSlider(value=3,min=1,max=5,step=1,description=self.chars[3],style=style)
        c5=IntSlider(value=3,min=1,max=5,step=1,description=self.chars[4],style=style)
        c6=IntSlider(value=3,min=1,max=5,step=1,description=self.chars[5],style=style)
        c7=IntSlider(value=3,min=1,max=5,step=1,description=self.chars[6],style=style)
        c8=IntSlider(value=3,min=1,max=5,step=1,description=self.chars[7],style=style)
        c9=IntSlider(value=3,min=1,max=5,step=1,description=self.chars[8],style=style)
        c10=IntSlider(value=3,min=1,max=5,step=1,description=self.chars[9],style=style)
        c11=IntSlider(value=3,min=1,max=5,step=1,description=self.chars[10],style=style)
        c12=IntSlider(value=3,min=1,max=5,step=1,description=self.chars[11],style=style)
        
        vbox1=VBox([c1,c2,c3,c4,c5,c6])
        vbox2=VBox([c7,c8,c9,c10,c11,c12])
               
        char_slider=HBox([vbox1,vbox2])
        
        mode=ToggleButtons(options=['전체','문장'])
        n=IntSlider(value=3,min=1,max=5,step=1,description='검색 수',style=style)

        button=Button(description="실행")
        output=Output()

        display(char_slider,mode,n,button,output)

        def on_button_clicked(b):
            with output:
                x=np.array([c1.value,c2.value,c3.value,c4.value,c5.value,c6.value,
                            c7.value,c8.value,c9.value,c10.value,c11.value,c12.value])
                if mode.value=='전체':
                    clear_output()
                    self.predict(x,n.value)
                elif mode.value=='문장':
                    clear_output()
                    self.predict_s(x,n.value)

        button.on_click(on_button_clicked)

