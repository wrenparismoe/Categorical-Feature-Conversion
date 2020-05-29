from modules.censusData import load_dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, SpatialDropout1D, Activation, concatenate
from keras.optimizers import Adam, SGD
from keras.layers.advanced_activations import ReLU, PReLU, LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from sklearn.tree import DecisionTreeClassifier

pd.options.display.max_columns = 999
pd.options.display.width = 999

features = []
numerical_features = []
categorical_features = []

def preprocessing():
    x_train, x_test, y_train, y_test = load_dataset()

    x_train = x_train.select_dtypes(include=[np.number]).astype(float).join(x_train.select_dtypes(exclude=[np.number]))
    x_test = x_test.select_dtypes(include=[np.number]).astype(float).join(x_test.select_dtypes(exclude=[np.number]))

    global features, numerical_features, categorical_features
    features = list(x_train.columns)
    numerical_features = list(x_train.select_dtypes(include=[float]).columns)
    categorical_features = [x for x in x_train.columns if x not in numerical_features]

    all_data = pd.concat([x_train, x_test])
    for c in categorical_features:
        le = LabelEncoder()
        all_data[c] = le.fit_transform(all_data[c])
    train_size = len(x_train)
    x_train = all_data.iloc[:train_size]
    x_test = all_data.iloc[train_size:]

    x_train_cat = np.array(x_train[categorical_features])
    x_test_cat = np.array(x_test[categorical_features])
    x_train_num = np.array(x_train[numerical_features])
    x_test_num = np.array(x_test[numerical_features])

    scalar = StandardScaler()
    x_train_num = scalar.fit_transform(x_train_num)
    x_test_num = scalar.fit_transform(x_test_num)
    return [x_train, y_train, x_test, y_test, x_train_cat, x_test_cat, x_train_num, x_test_num, all_data]


class Wide_and_Deep:
    def __init__(self, mode='wide and deep'):
        self.mode = mode
        x_train, y_train, x_test, y_test, x_train_cat, x_test_cat, x_train_num, x_test_num, all_data = preprocessing()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_train_cat = x_train_cat
        self.x_test_cat = x_test_cat
        self.x_train_num = x_train_num
        self.x_test_num = x_test_num
        self.all_data = all_data
        self.poly = PolynomialFeatures(degree=2, interaction_only=True)

        self.x_train_cat_poly = self.poly.fit_transform(x_train_cat)
        self.x_test_cat_poly = self.poly.transform(x_test_cat)
        self.categ_inputs = None
        self.conti_input = None
        self.deep_component_outlayer = None
        self.logistic_input = None
        self.model = None
        self.x_train_enc = None
        self.x_test_enc = None

    def deep_component(self):
        categ_inputs = []
        categ_embeds = []

        for i in range(len(categorical_features)):
            input_i = Input(shape=(1,), dtype='int32')
            dim = len(np.unique(self.all_data[categorical_features[i]]))
            embed_dim = int(np.ceil(dim ** 0.25))
            embed_i = Embedding(dim, embed_dim, input_length=1)(input_i)
            flatten_i = Flatten()(embed_i)
            categ_inputs.append(input_i)
            categ_embeds.append(flatten_i)

        conti_input = Input(shape=(len(numerical_features),))
        conti_dense = Dense(256, use_bias=False)(conti_input)
        # 全結合層と各Embeddingの出力をくっつける
        concat_embeds = concatenate([conti_dense] + categ_embeds)
        concat_embeds = Activation('relu')(concat_embeds)
        bn_concat = BatchNormalization()(concat_embeds)

        fc1 = Dense(512, use_bias=False)(bn_concat)
        ac1 = ReLU()(fc1)
        bn1 = BatchNormalization()(ac1)
        fc2 = Dense(256, use_bias=False)(bn1)
        ac2 = ReLU()(fc2)
        bn2 = BatchNormalization()(ac2)
        fc3 = Dense(128)(bn2)
        ac3 = ReLU()(fc3)

        self.categ_inputs = categ_inputs
        self.conti_input = conti_input
        self.deep_component_outlayer = ac3

    def wide_component(self):
        # カテゴリーデータだけ線形モデルに入れる
        dim = self.x_train_cat_poly.shape[1]
        self.logistic_input = Input(shape=(dim,))


    def create_model(self):
        self.deep_component()
        self.wide_component()
        if self.mode == 'wide and deep':
            out_layer = concatenate([self.deep_component_outlayer, self.logistic_input])
            inputs = [self.conti_input] + self.categ_inputs + [self.logistic_input]
        elif self.mode == 'deep':
            out_layer = self.deep_component_outlayer
            inputs = [self.conti_input] + self.categ_inputs
        else:
            print('wrong mode')
            return

        output = Dense(1, activation='sigmoid')(out_layer)
        self.model = Model(inputs=inputs, outputs=output)

    def train_model(self, epochs=15, optimizer='adam', batch_size=128):
        if not self.model:
            print('You have to create model first')
            return

        if self.mode == 'wide and deep':
            input_data = [self.x_train_num] + \
                         [self.x_train_cat[:, i] for i in range(self.x_train_cat.shape[1])] + \
                         [self.x_train_cat_poly]
        elif self.mode == 'deep':
            input_data = [self.x_train_num] + \
                         [self.x_train_cat[:, i] for i in range(self.x_train_cat.shape[1])]
        else:
            print('wrong mode')
            return

        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(input_data, self.y_train, epochs=epochs, batch_size=batch_size)

    def evaluate_model(self):
        if not self.model:
            print('You have to create model first')
            return

        if self.mode == 'wide and deep':
            input_data = [self.x_test_num] + \
                         [self.x_test_cat[:, i] for i in range(self.x_test_cat.shape[1])] + \
                         [self.x_test_cat_poly]
        elif self.mode == 'deep':
            input_data = [self.x_test_num] + \
                         [self.x_test_cat[:, i] for i in range(self.x_test_cat.shape[1])]
        else:
            print('wrong mode')
            return

        loss, acc = self.model.evaluate(input_data, self.y_test)
        print(f'test_loss: {loss} - test_acc: {acc}')

    def save_model(self, filename='wide_and_deep.h5'):
        self.model.save(filename)


if __name__ == '__main__':
    wide_deep_net = Wide_and_Deep()
    wide_deep_net.create_model()
    wide_deep_net.train_model()
    wide_deep_net.evaluate_model()
    wide_deep_net.save_model()
    #plot_model(wide_deep_net.model, to_file='model.png', show_shapes=True, show_layer_names=False)









