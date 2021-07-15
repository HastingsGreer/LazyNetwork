import torch
import copy

class LazyNetworkComponent:
    def __init__(self):
        self.requestedRegion = None
    def updateRequestedRegion(self, region):
        
        if not(self.requestedRegion):
            self.requestedRegion = copy.deepcopy(region)
        else:
            for current_interval, new_interval in zip(self.requestedRegion, region):
                current_interval[0] = min(current_interval[0], new_interval[0])
                current_interval[1] = max(current_interval[1], new_interval[1])
        self.updateUpstreamRequestedRegion(self.requestedRegion)
        for input_ in self.inputs:
            input_.updateRequestedRegion(self.upstream_region)
    def updateUpstreamRequestedRegion(self, region):
        raise(NotImplementedError)
        

class LazyInput(LazyNetworkComponent):
    def __init__(self):
        super(LazyInput, self).__init__()
        self.inputs = []
    def updateUpstreamRequestedRegion(self, region):
        pass 
class LazyConvolution(LazyNetworkComponent):
    def __init__(self, conv, input_):
        super(LazyConvolution, self).__init__()
        self.component = conv
        self.inputs = [input_]

    def updateUpstreamRequestedRegion(self, region):
        self.upstream_region = [[a - 1, b + 1] for a, b in region]
class LazyConcatenate(LazyNetworkComponent):
    def __init__(self, concat, inputs):
        super(LazyConcatenate, self).__init__()
        self.component = concat
        self.inputs = inputs

    def updateUpstreamRequestedRegion(self, region):
        self.upstream_region = region
class LazyPooling(LazyNetworkComponent):
    
    def __init__(self, pool, input_):
        super(LazyPooling, self).__init__()
        self.component = pool
        self.inputs = [input_]
    def updateUpstreamRequestedRegion(self, region):
        self.upstream_region = [[a*2, b*2] for a, b in region]
class LazyUpsample(LazyNetworkComponent):
    def __init__(self, upsample, input_):
        super(LazyUpsample, self).__init__()
        self.component = upsample
        self.inputs = [input_]
    def updateUpstreamRequestedRegion(self, region):
        self.upstream_region = []
        for low, hi in region:
            if low %2 != 0:
                low -= 1
            if hi %2 != 0:
                hi += 1
            self.upstream_region.append([low //2 , hi // 2])
if __name__ == "__main__":
    input_ = LazyInput()
    component = "dummy"

    c1a = LazyConvolution(component, input_)
    c1b = LazyConvolution(component, c1a)
    p2 = LazyPooling(component, c1b)
    c2a = LazyConvolution(component, p2)
    c2b = LazyConvolution(component, c2a)
    p3 = LazyPooling(component, c2b)
    c3a = LazyConvolution(component, p3)
    c3b = LazyConvolution(component, c3a)
    p4 = LazyPooling(component, c3b)
    c4a = LazyConvolution(component, p4)
    c4b = LazyConvolution(component, c4a)
    p5 = LazyPooling(component, c4b)
    c5a = LazyConvolution(component, p5)
    c5b = LazyConvolution(component, c5a)
    u4 = LazyUpsample(component, c5b)
    cn4 = LazyConcatenate(component, [c4b, u4])
    c4c = LazyConvolution(component, cn4)
    c4d = LazyConvolution(component, c4c)
    u3 = LazyUpsample(component, c4d)
    cn3 = LazyConcatenate(component, [c3b, u3])
    c3c = LazyConvolution(component, cn3)
    c3d = LazyConvolution(component, c3c)
    u2 = LazyUpsample(component, c3d)
    cn2 = LazyConcatenate(component, [c2b, u2])
    c2c = LazyConvolution(component, cn2)
    c2d = LazyConvolution(component, c2c)
    u1 = LazyUpsample(component, c2d)
    cn1 = LazyConcatenate(component, [c1b, u1])
    c1c = LazyConvolution(component, cn1)
    c1d = LazyConvolution(component, c1c)

    c1d.updateRequestedRegion([[-2, 386], [-2, 386]])


    inv = input_
    print(inv.requestedRegion)
    print(inv.requestedRegion[0][1] - inv.requestedRegion[0][0])

"""
if __name__ == "__main__":
    conv = "dummy"
    input_ = LazyInput()
    l1 = LazyConvolution(conv, input_)
    l2 = LazyConvolution(conv, l1)
    l3 = LazyConvolution(conv, l2)

    l3.updateRequestedRegion([[0, 32], [64, 96]])

    print(input_.requestedRegion)
   """


