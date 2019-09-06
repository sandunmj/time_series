import matplotlib.pyplot as plt
import json
import requests


def show_metrics(history):
    plt.plot(history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.plot(history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


url_graph = 'http://127.0.0.1:1111/history'
req_graph = requests.get(url_graph)
show_metrics(req_graph.json())
