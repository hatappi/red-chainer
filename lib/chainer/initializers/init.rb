module Chainer
  module Initializers
    def self.generate_array(initializer, shape, xm: Chainer.get_default_device.xm)
      klass = xm::SFloat
      if initializer.respond_to?(:dtype) && initializer.dtype
        klass = initializer.dtype
      end
      array = klass.new(shape).rand
      initializer.(array)
    end

    def self.get_initializer(initializer, xm: Chainer.get_default_device.xm)
      return HeNormal.new(scale: 1 / xm::NMath.sqrt(2)) if initializer.nil?
      return Constant.new(initializer) if initializer.kind_of?(Numeric)
      return Constant.new(initializer) if Chainer.array?(initializer)

      unless initializer.respond_to?(:call)
        raise TypeError, "invalid type of initializer: #{initializer.class}"
      end

      return initializer
    end

    def self.nan(dtype: nil)
      Constant.new(Float::NAN, dtype: dtype)
    end
  end
end
