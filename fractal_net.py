import six
import numpy as np
import chainer.links as L
import chainer.functions as F
import nutszebra_chainer
import functools
from collections import defaultdict


class Conv_BN_ReLU(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(Conv_BN_ReLU, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel, filter_size, stride, pad),
            bn=L.BatchNormalization(out_channel),
        )

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def __call__(self, x, train=False, dropout_ratio=0.0):
        if type(x) is int:
            return 0
        return F.relu(self.bn(self.conv(F.dropout(x, ratio=dropout_ratio, train=train)), test=not train))

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)


class f1(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1), dropout_ratio=0.0):
        super(f1, self).__init__()
        modules = []
        modules.append(('conv_bn_relu', Conv_BN_ReLU(in_channel, out_channel, filter_size=filter_size, stride=stride, pad=pad)))
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.dropout_ratio = dropout_ratio
        self.iamf1 = True

    def weight_initialization(self):
        return self.conv_bn_relu.weight_initialization()

    def __call__(self, x, train=False):
        return self.conv_bn_relu(x, train=train, dropout_ratio=self.dropout_ratio)

    def count_parameters(self):
        return self.conv_bn_relu.count_parameters()


class FractalExpansion(nutszebra_chainer.Model):

    def __init__(self, f_1, f_2, f_name='f2', filter_size=(3, 3), stride=(1, 1), pad=(1, 1), dropout_ratio=0.0):
        super(FractalExpansion, self).__init__()
        modules = []
        modules.append(('f1', f1(f_1.in_channel, f_2.out_channel, filter_size=filter_size, stride=stride, pad=pad, dropout_ratio=dropout_ratio)))
        modules.append(('{}_1'.format(f_name), f_1))
        modules.append(('{}_2'.format(f_name), f_2))
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.f_name = f_name
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.dropout_ratio = dropout_ratio
        self.in_channel = f_1.in_channel
        self.out_channel = f_2.out_channel
        self.student = True
        self.teacher = True
        self.iamf1 = False

    def weight_initialization(self):
        self['{}_1'.format(self.f_name)].weight_initialization()
        self['{}_2'.format(self.f_name)].weight_initialization()
        self['f1'].weight_initialization()

    def __call__(self, x, train=False):
        if self.student is True and self.teacher is True:
            h = self['{}_1'.format(self.f_name)](x, train=train)
            h = self['{}_2'.format(self.f_name)](h, train=train)
            return (self['f1'](x, train=train) + h) / 2
        elif self.student is True and self.teacher is False:
            return self['f1'](x, train=train)
        elif self.student is False and self.teacher is True:
            h = self['{}_1'.format(self.f_name)](x, train=train)
            h = self['{}_2'.format(self.f_name)](h, train=train)
            return h
        return 0

    def count_parameters(self):
        count = 0
        count += self['{}_1'.format(self.f_name)].count_parameters()
        count += self['{}_2'.format(self.f_name)].count_parameters()
        count += self['f1'].count_parameters()
        return count


def generate_fractal_block(c, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1), dropout_ratio=0.0):
    if int(c) == 1:
        return f1(in_channel, out_channel, filter_size=filter_size, stride=stride, pad=pad, dropout_ratio=dropout_ratio)
    f_1 = generate_fractal_block(c - 1, in_channel, out_channel, filter_size=filter_size, stride=stride, pad=pad, dropout_ratio=dropout_ratio)
    f_2 = generate_fractal_block(c - 1, out_channel, out_channel, filter_size=filter_size, stride=stride, pad=pad, dropout_ratio=dropout_ratio)
    return FractalExpansion(f_1, f_2, f_name='f{}'.format(int(c)), filter_size=filter_size, stride=stride, pad=pad, dropout_ratio=dropout_ratio)


class FractalBlock(nutszebra_chainer.Model):

    def __init__(self, c, in_channel, out_channel, droppath_local=0.5, local_probability=0.15, dropout_ratio=0.0):
        super(FractalBlock, self).__init__()
        modules = []
        modules += [('block', generate_fractal_block(c, in_channel, out_channel, dropout_ratio=dropout_ratio))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.c = c
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.dropout_ratio = dropout_ratio
        self.droppath_local = droppath_local
        self.local_probability = local_probability

    @staticmethod
    def _children(ele):
        return hasattr(ele, '_children')

    @staticmethod
    def _teacher_or_student(ele):
        return hasattr(ele, 'teacher') or hasattr(ele, 'student')

    @staticmethod
    def _f1(ele):
        return hasattr(ele, 'iamf1')

    @staticmethod
    def _find(model):
        links = {}

        def dfs(name, ele):
            if FractalBlock._f1(ele) is True:
                if ele.iamf1 is False:
                    # model/f2_2/ -> model/f2_2
                    links[name[:-1]] = ele
                    for child in ele._children:
                        dfs(name + child + '/', ele[child])
        dfs('model/', model)
        return links

    @staticmethod
    def recursive_reference(obj, keys):
        for key in keys:
            obj = obj[key]
        return obj

    @staticmethod
    def detect_fractal_expansion(obj):
        return FractalBlock._teacher_or_student(obj) and FractalBlock._f1(obj)

    @staticmethod
    def yield_fractal_expansion(fractal_block):
        fractal_components = FractalBlock._find(fractal_block)
        for key, obj in fractal_components.items():
            if FractalBlock.detect_fractal_expansion(obj) is True:
                yield key, obj

    @staticmethod
    def show_droppath(fractal_block):
        for key, obj in FractalBlock.yield_fractal_expansion(fractal_block):
            t = obj.teacher
            s = obj.student
            print('{}: teacher->{}, student->{}'.format(key, t, s))

    @staticmethod
    def drop_all(fractal_block):
        for key, obj in FractalBlock.yield_fractal_expansion(fractal_block):
            obj.teacher = False
            obj.student = False

    @staticmethod
    def revive_all(fractal_block):
        for key, obj in FractalBlock.yield_fractal_expansion(fractal_block):
            obj.teacher = True
            obj.student = True

    @staticmethod
    def get_C(fractal_block):
        C = 0
        for key, obj in FractalBlock.yield_fractal_expansion(fractal_block):
            C = np.max([C, len(key.split('/'))])
        return C + 1

    @staticmethod
    def get_same_depth(fractal_block, depth):
        answer = {}
        for key, obj in FractalBlock.yield_fractal_expansion(fractal_block):
            if depth == len(key.split('/')):
                answer[key] = obj
        return answer.items()

    @staticmethod
    def global_droppath(fractal_block):
        FractalBlock.drop_all(fractal_block)
        C = FractalBlock.get_C(fractal_block)
        path = np.random.randint(1, C + 1)
        if path == C:
            for key, obj in FractalBlock.yield_fractal_expansion(fractal_block):
                obj.teacher = True
            return '{}: {} teacher'.format(C, path)
        obj = [fractal_block]
        keep_path = path
        while 1:
            if path == 1:
                break
            new_obj = []
            for o in obj:
                o.teacher = True
                if FractalBlock.detect_fractal_expansion(o) is True:
                    tmp = o._children[:]
                    tmp.remove('f1')
                    new_obj += [o[name] for name in tmp]
            obj = new_obj
            path -= 1
        path = keep_path
        for key, obj in FractalBlock.get_same_depth(fractal_block, path):
            obj.student = True
        return '{}: {} student'.format(C, path)

    @ staticmethod
    def path_is_alive(obj):
        return obj.teacher or obj.student

    @staticmethod
    def reachable(fractal_block):
        for key, obj in FractalBlock._find(fractal_block).items():
            key = key.split('/')
            while 1:
                if FractalBlock.path_is_alive(FractalBlock.recursive_reference(fractal_block, key[1:])) is False:
                    break
                if len(key) == 1:
                    return True
                key = key[:-1]
        return False

    @staticmethod
    def prune(fractal_block):
        C = FractalBlock.get_C(fractal_block)
        for i in six.moves.range(C):
            for key, obj in FractalBlock._find(fractal_block).items():
                key = key.split('/')
                # if teacher is dead, then children are all dead
                obj = FractalBlock.recursive_reference(fractal_block, key[1:])
                if obj.teacher is False and len(key) >= 2:
                    for name, obj in FractalBlock.yield_fractal_expansion(FractalBlock.recursive_reference(fractal_block, key[1:])):
                        if not name == 'model':
                            obj.teacher = False
                            obj.student = False
                # student and teacher are both dead, next guy is dead
                # and the teacher of above guy is dead
                if FractalBlock.path_is_alive(FractalBlock.recursive_reference(fractal_block, key[1:])) is False:
                    obj = FractalBlock.recursive_reference(fractal_block, key[1:-1] + FractalBlock.the_other(key[-1]))
                    obj.teacher = False
                    obj.student = False
                    obj = FractalBlock.recursive_reference(fractal_block, key[1:-1])
                    obj.teacher = False

    @staticmethod
    def the_other(key):
        if key == 'model':
            return []
        if key[-1] == '1':
            return [key[:-1] + '2']
        else:
            return [key[:-1] + '1']

    @staticmethod
    def _local_droppath(fractal_block, local_probability):
        for key, obj in FractalBlock.yield_fractal_expansion(fractal_block):
            if np.random.random() <= local_probability:
                obj.teacher = False
            if np.random.random() <= local_probability:
                obj.student = False

    @staticmethod
    def local_droppath(fractal_block, local_probability=0.15):
        while 1:
            FractalBlock.revive_all(fractal_block)
            FractalBlock._local_droppath(fractal_block, local_probability=local_probability)
            FractalBlock.prune(fractal_block)
            if FractalBlock.reachable(fractal_block) is True:
                break

    def drop_path(self, fractal_block):
        global_ratio = 1.0 - self.droppath_local
        if np.random.rand() <= global_ratio:
            self.global_droppath(self.block)
        else:
            self.local_droppath(self.block, self.local_probability)

    def weight_initialization(self):
        self.block.weight_initialization()

    def __call__(self, x, train=False):
        if train is True:
            self.drop_path(self.block)
        else:
            self.revive_all(self.block)
        return self.block(x, train=train)

    def count_parameters(self):
        return self.block.count_parameters()


class FractalNet(nutszebra_chainer.Model):

    def __init__(self, num_category, block_num=5, C=(3, 3, 3, 3, 3), channels=(64, 128, 256, 512, 512), block_dropout=(0.0, 0.1, 0.2, 0.3, 0.4)):
        super(FractalNet, self).__init__()
        modules = []
        in_channel = 3
        for i, c, out_channel, dropout_ratio in six.moves.zip(six.moves.range(1, block_num + 1), C, channels, block_dropout):
            modules += [('fractal_block{}'.format(i), FractalBlock(c, in_channel, out_channel, dropout_ratio=dropout_ratio))]
            in_channel = out_channel
        modules += [('conv_bn_relu', Conv_BN_ReLU(out_channel, num_category, filter_size=(1, 1), stride=(1, 1), pad=(0, 0)))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.num_category = num_category
        self.block_num = block_num
        self.channels = channels
        self.block_dropout = block_dropout
        self.name = 'fractal_net_{}'.format(num_category)

    def weight_initialization(self):
        for name, link in self.modules:
            link.weight_initialization()

    def __call__(self, x, train=False):
        for i in six.moves.range(1, self.block_num + 1):
            x = self['fractal_block{}'.format(i)](x, train=train)
            x = F.max_pooling_2d(x, ksize=(2, 2), stride=(2, 2), pad=(0, 0))
        h = self.conv_bn_relu(x, train=train, dropout_ratio=0.0)
        num, categories, y, x = h.data.shape
        h = F.reshape(F.average_pooling_2d(h, (y, x)), (num, categories))
        return h

    def count_parameters(self):
        count = 0
        for name, link in self.modules:
            count += link.count_parameters()
        return count

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy
