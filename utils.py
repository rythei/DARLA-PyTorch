import math

def output_after_conv(input_size, kernel_size, padding, stride):
    W = input_size
    K = kernel_size
    S = stride
    P = padding

    O = (W-K+2*P)/float(S) + 1
    return int(math.ceil(O))


def same_padding(image_dim, kernel_size, stride):
    K = kernel_size
    W = image_dim
    S = stride
    return int(math.ceil((S*(W-1) - W + K)/2))


if __name__ == '__main__':
    ## test if padding works ##
    image_dim = 64
    kernel_size = 4
    stride = 2
    padding = same_padding(image_dim, kernel_size, stride)

    print('Same padding works: ', output_after_conv(image_dim, kernel_size, padding, stride) == image_dim)