
from model import MNIST_Model, ISOLET_Model

cpuPower = 45 # in Watts, replace with your CPU TDP

mnistModel = MNIST_Model(10000, 2, 2, 28)
isoletModel = ISOLET_Model(10000, 2, 2)

for name, model in [("MNIST", mnistModel), ("ISOLET", isoletModel)]:
    # trivial training
    #model.trainOneIteration()
    avgEncodeLatency, avgClassifyLatency = model.benchmark(nTests=1000)
    encodeThroughput, classifyThroughput = model.benchThroughput()

    totalThroughput = 1 / (1/encodeThroughput + 1/classifyThroughput)
    encodeEnergy = cpuPower / encodeThroughput
    classifyEnergy = cpuPower / classifyThroughput

    print(f"Benchmark results for {name} model:")

    print(f"\tEncode latency: {1000*avgEncodeLatency:.4f}ms/input")
    print(f"\tClassify latency: {1000*avgClassifyLatency:.4f}ms/input")
    print(f"\tTotal throughput (all cores): {totalThroughput:.2f} inputs/s")
    print(f"\tEncode energy: {encodeEnergy*1000:.4f}mJ/input")
    print(f"\tClassify energy: {classifyEnergy*1000:.4f}mJ/input")
    print()



