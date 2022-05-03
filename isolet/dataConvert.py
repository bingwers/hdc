
def mapFloatToUint8(f):
    i = int((f + 1) / 2 * 255)
    if i > 255:
        i = 255
    if i < 0:
        i = 0
    return i

def convert(fn, outName):
    with open(fn) as fp:
        lines = fp.readlines()

    featuresFp = open(outName + "-features.idx3-ubyte", 'wb')
    labelsFp = open(outName + "-labels.idx1-ubyte", 'wb')
    
    featureSize = len(lines[0].split(',')) - 1
    nItems = len(lines)

    featuresFp.write((0x0308).to_bytes(4, 'big'))
    featuresFp.write(nItems.to_bytes(4, 'big'))
    featuresFp.write(featureSize.to_bytes(4, 'big'))
    featuresFp.write((1).to_bytes(4, 'big'))

    labelsFp.write((0x0108).to_bytes(4, 'big'))
    labelsFp.write(nItems.to_bytes(4, 'big'))

    for line in lines:
        entries = line.split(',')
        entries, label = entries[:-1], entries[-1]

        label = int(label.strip()[:-1]) - 1
        label = label.to_bytes(1, 'little')

        entries = [float(entry.strip()) for entry in entries]
        entries = [mapFloatToUint8(entry) for entry in entries]
        entries = bytes(entries)

        featuresFp.write(entries)
        labelsFp.write(label)

    featuresFp.close()
    labelsFp.close()

convert("isolet1+2+3+4.data", 'train')
convert("isolet5.data", 'test')