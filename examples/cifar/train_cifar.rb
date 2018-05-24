require 'chainer'
# require __dir__ + '/models/vgg'
require __dir__ + '/models/resnet18'
require 'optparse'

args = {
  dataset: 'cifar10',
  frequency: -1,
  batchsize: 64,
  learnrate: 0.05,
  epoch: 300,
  out: 'result',
  resume: nil
}


opt = OptionParser.new
opt.on('-d', '--dataset VALUE', "The dataset to use: cifar10 or cifar100 (default: #{args[:dataset]})") { |v| args[:dataset] = v }
opt.on('-b', '--batchsize VALUE', "Number of images in each mini-batch (default: #{args[:batchsize]})") { |v| args[:batchsize] = v.to_i }
opt.on('-f', '--frequency VALUE', "Frequency of taking a snapshot (default: #{args[:frequency]})") { |v| args[:frequency] = v.to_i }
opt.on('-l', '--learnrate VALUE', "Learning rate for SGD (default: #{args[:learnrate]})") { |v| args[:learnrate] = v.to_f } 
opt.on('-e', '--epoch VALUE', "Number of sweeps over the dataset to train (default: #{args[:epoch]})") { |v| args[:epoch] = v.to_i }
opt.on('-o', '--out VALUE', "Directory to output the result (default: #{args[:out]})") { |v| args[:out] = v }
opt.on('-r', '--resume VALUE', "Resume the training from snapshot") { |v| args[:resume] = v }
opt.parse!(ARGV)

# Set up a neural network to train.
# Classifier reports softmax cross entropy loss and accuracy at every
# iteration, which will be used by the PrintReport extension below.
if args[:dataset] == 'cifar10'
    puts 'Using CIFAR10 dataset.'
    class_labels = 10
    train, test = Chainer::Datasets::CIFAR.get_cifar10
elsif args[:dataset] == 'cifar100'
    puts 'Using CIFAR100 dataset.'
    class_labels = 100
    train, test = Chainer::Datasets::CIFAR.get_cifar100
else
    raise 'Invalid dataset choice.'
end

puts "setup..."

model = Chainer::Links::Model::Classifier.new(ResNet18.new(n_classes: class_labels))

optimizer = Chainer::Optimizers::MomentumSGD.new(lr: args[:learnrate])
optimizer.setup(model)

train_iter = Chainer::Iterators::SerialIterator.new(train, args[:batchsize])
test_iter = Chainer::Iterators::SerialIterator.new(test, args[:batchsize], repeat: false, shuffle: false)

updater = Chainer::Training::StandardUpdater.new(train_iter, optimizer, device: -1)
trainer = Chainer::Training::Trainer.new(updater, stop_trigger: [args[:epoch], 'epoch'], out: args[:out])

trainer.extend(Chainer::Training::Extensions::Evaluator.new(test_iter, model, device: -1))

trainer.extend(Chainer::Training::Extensions::ExponentialShift.new('lr', 0.5), trigger: [25, 'epoch'])

frequency = args[:frequency] == -1 ? args[:epoch] : [1, args[:frequency]].max
trainer.extend(Chainer::Training::Extensions::Snapshot.new, trigger: [frequency, 'epoch'])

trainer.extend(Chainer::Training::Extensions::LogReport.new)    
trainer.extend(Chainer::Training::Extensions::PrintReport.new(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(Chainer::Training::Extensions::ProgressBar.new)

if args[:resume]
  Chainer::Serializers::MarshalDeserializer.load_file(args[:resume], trainer)
end

trainer.run

