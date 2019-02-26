module Chainer
  module Functions
    module Connection
      class Deconvolution2DFunction < Chainer::FunctionNode
        def self.deconvolution_2d(x, w, b: nil, stride: 1, pad: 0, outsize: nil)
          func = Deconvolution2DFunction.new(stride: stride, pad: pad, outsize: outsize)
          if b.nil?
            args = x, w
          else
            args = x, w, b
          end
          func.apply(args).first
        end

        def initialize(stride: 1, pad: 0, outsize: nil)
          @sy, @sx = stride.is_a?(::Array) ? stride : [stride, stride]
          @ph, @pw = pad.is_a?(::Array) ? pad : [pad, pad]
          @outh, @outw = outsize.nil? ? [nil, nil] : outsize
        end

        def forward(inputs)
          retain_inputs([0, 1])
          x, w = inputs[0...2]
          b = inputs.size == 3 ? inputs[2] : nil

          unless inputs.all? { |i| i.is_a?(Numo::NArray) }
            if b.nil?
              raise TypeError, "Numo::NArray must not be used together w: #{w.class}, x: #{x.class}"
            else
              raise TypeError, "Numo::NArray must not be used together w: #{w.class}, x: #{x.class}, b: #{b.class}"
            end
          end

          kh, kw = w.shape[2..-1]
          _, _, x_h, x_w = x.shape

          gcol = Chainer::Utils::Math.tensordot(w, x, [0, 1]).cast_to(x.class)
          # - k, m, n: shape of out_channel
          # - b: number of inputs
          # - h, w: height and width of kernels
          # k, m, n, b, h, w -> b, k, m, n, h, w
          gcol = gcol.transpose(3, 0, 1, 2, 4, 5)

          if @outh.nil?
            @outh = Chainer::Utils::Conv.get_deconv_outsize(x_h, kh, @sy, @ph)
            raise TypeError, 'Height in the output should be positive.' if @outh <= 0
          end
          if @outw.nil?
            @outw = Chainer::Utils::Conv.get_deconv_outsize(x_w, kw, @sx, @pw)
            raise TypeError, 'Width in the output should be positive.' if @outw <= 0
          end

          y = Chainer::Utils::Conv.col2im(gcol, @sy, @sx, @ph, @pw, @outh, @outw)
          if !b.nil?
            y += b.reshape(1, b.size, 1, 1)
          end
          [y]
        end
      end
    end
  end
end
