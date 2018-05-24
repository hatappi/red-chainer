class BottleNeck < Chainer::Chain
  include Chainer::Functions::Activation
  include Chainer::Initializers
  include Chainer::Links::Connection
  include Chainer::Links::Normalization

  def initialize(n_in, n_mid, n_out, stride=1, use_conv=false)
    super()
    w = HeNormal.new

    init_scope do
      @conv1 = Convolution2D.new(n_in, n_mid, 1, stride: stride, pad: 0, nobias: true, initial_w: w)
      @bn1 = BatchNormalization.new(n_mid)
      @conv2 = Convolution2D.new(n_mid, n_mid, 3, stride: 1, pad: 1, nobias: true, initial_w: w)
      @bn2 = BatchNormalization.new(n_mid)
      @conv3 = Convolution2D.new(n_mid, n_out, 1, stride: 1, pad: 0, nobias: true, initial_w: w)
      @bn3 = BatchNormalization.new(n_out)
      if use_conv
        @conv4 = Convolution2D.new(n_in, n_out, 1, stride: stride, pad: 0, nobias: true, initial_w: w)
        @bn4 = BatchNormalization.new(n_out)
      end
    end
    @use_conv = use_conv
  end

  def call(x)
    h = Relu.relu(@bn1.(@conv1.(x)))
    h = Relu.relu(@bn2.(@conv2.(h)))
    h = @bn3.(@conv3.(h))
    if @use_conv
      h = @bn4.(@conv4.(x))
    else
      h + x
    end
  end

end


class BottleNeckA < Chainer::Chain
  include Chainer::Functions::Activation
  include Chainer::Initializers
  include Chainer::Links::Connection
  include Chainer::Links::Normalization

  def initialize(in_size, ch, out_size, stride=2)
    super()
    initial_w = HeNormal.new

    init_scope do
      @conv1 = Convolution2D.new(in_size, ch, 1, stride: stride, pad: 0, initial_w: initial_w, nobias: true)
      @bn1 = BatchNormalization.new(ch)
      @conv2 = Convolution2D.new(ch, ch, 3, stride: 1, pad: 1, initial_w: initial_w, nobias: true)
      @bn2 = BatchNormalization.new(ch)
      @conv3 = Convolution2D.new(ch, out_size, 1, stride: 1, pad: 0, initial_w: initial_w, nobias: true)
      @bn3 = BatchNormalization.new(out_size)
      @conv4 = Convolution2D.new(in_size, out_size, 1, stride: stride, pad: 0, initial_w: initial_w, nobias: true)
      @bn4 = BatchNormalization.new(out_size)
    end
  end

  def call(x)
    h1 = Relu.relu(@bn1.(@conv1.(x)))
    h1 = Relu.relu(@bn2.(@conv2.(h1)))
    h1 = @bn3.(@conv3.(h1))
    h2 = @bn4.(@conv4.(x))

    Relu.relu(h1 + h2)
  end
end

class BottleNeckB < Chainer::Chain
  include Chainer::Functions::Activation
  include Chainer::Initializers
  include Chainer::Links::Connection
  include Chainer::Links::Normalization

  def initialize(in_size, ch)
    super()
    initial_w = HeNormal.new

    init_scope do
      @conv1 = Convolution2D.new(in_size, ch, 1, stride: 1, pad: 0, initial_w: initial_w, nobias: true)
      @bn1 = BatchNormalization.new(ch)
      @conv2 = Convolution2D.new(ch, ch, 3, stride: 1, pad: 1, initial_w: initial_w, nobias: true)
      @bn2 = BatchNormalization.new(ch)
      @conv3 = Convolution2D.new(ch, in_size, 1, stride: 1, pad: 0, initial_w: initial_w, nobias: true)
      @bn3 = BatchNormalization.new(in_size)
    end
  end

  def call(x)
    h = Relu.relu(@bn1.(@conv1.(x)))
    h = Relu.relu(@bn2.(@conv2.(h)))
    h = @bn3.(@conv3.(h))

    Relu.relu(h + x)
  end
end

class Block < Chainer::ChainList
  def initialize(layer, in_size, ch, out_size, stride=2)
    super()
    add_link(BottleNeck.new(in_size, ch, out_size, stride, true))
    (layer-1).times do
      add_link(BottleNeck.new(out_size, ch, out_size))
    end
  end

  def call(x)
    @children.each do |f|
      x = f.(x)
    end
    x
  end
end

class ResNet50 < Chainer::Chain
  include Chainer::Functions::Activation
  include Chainer::Initializers
  include Chainer::Links::Connection
  include Chainer::Links::Normalization
  include Chainer::Functions::Pooling
  include Chainer::Functions::Loss
  include Chainer::Functions::Evaluation

  INSIZE = 224

  def initialize(n_classes: 10, n_layers: [3, 4, 6, 3])
    super()
    initial_w = HeNormal.new

    init_scope do
      @conv = Convolution2D.new(nil, 64, 3, stride: 1, pad: 0, nobias: true, initial_w: initial_w)
      @bn = BatchNormalization.new(64)

      @res2 = Block.new(n_layers[0], 64, 64, 256, 1)
      @res3 = Block.new(n_layers[1], 256, 128, 512, 2)
      @res4 = Block.new(n_layers[2], 512, 256, 1024, 2)
      @res5 = Block.new(n_layers[3], 1024, 512, 2048, 2)
      @fc = Linear.new(nil, out_size: n_classes)
    end
  end

  def call(x)
    h = Relu.relu(@bn.(@conv.(x)))
    # h = MaxPooling2D.max_pooling_2d(Relu.relu(h), 3, stride: 2)
    h = @res2.(h)
    h = @res3.(h)
    h = @res4.(h)
    h = @res5.(h)
    puts ">>>>>>>>>>"
    h = AveragePooling2D.average_pooling_2d(h, h.shape[2..-1])
    @fc.(h)
  end
end
