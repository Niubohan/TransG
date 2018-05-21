from TransG import TransG
import time

if __name__ == '__main__':
    dir = "D:\\Data\\Knowledge Embedding\\EM\\"
    start_time = time.time()
    model = TransG(dir, 50, 0.001, 3.5, 4, 0.01)
    model.train(1)
    end_time = time.time()
    model.test(1)
    model.save()
    model.draw(1000)
    print (end_time - start_time)
    print "OK"
