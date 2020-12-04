from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs=["너무 재밌어요", "참 최고예요", "참 잘 만든 영화예요", "추천하고 싶은 영화입니다.", "한 번 더 보고 싶네요", "글쎄요",
        "별로예요", "생각보다 지루해요", "연기가 어색해요", "재미없어요", "너무 재미없다", "참 재밌네요"]

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x=token.texts_to_sequences(docs)
print(x)

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x=pad_sequences(x, padding='pre') # 뒤 : post
print(pad_x) 
print(pad_x.shape) #(12, 5)

word_size=len(token.word_index) + 1
print("전체 토큰 사이즈 : ", word_size) # 25

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D

model=Sequential()
# Embedding(단어사전의 갯수, 아웃풋 노드의 갯수, 컬럼 수) ===> 원핫인코딩을 하면 (25,25) 였을 아웃풋을 (25, 10)으로 벡터화된 단어사전을 만듦
# 단어사전의 갯수를 더 큰 값을 넣는 건 상관 없지만 적은 값을 넣으면 메모리가 터진다.
model.add(Embedding(25, 10, input_length=5))  # => 컬럼의 갯수를 다르게 넣어도 돌아는 감.
# model.add(Embedding(25, 10)) # => 컬럼을 표기하지 않아도 연산은 같음. summaray에서 표시 안됨. 
model.add(LSTM(32))
# model.add(Conv1D(32, 2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=30)
acc=model.evaluate(pad_x, labels)[1]

print('acc', acc)