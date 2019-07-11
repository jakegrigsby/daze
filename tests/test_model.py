import pytest

import deepzip as dz

def test_model_with_tf_dataset():
    model = dz.Model(dz.nets.encoders.Encoder_32x32(), dz.nets.decoders.Decoder_32x32())
    x_train, x_val = dz.data.cifar10.load(64, 'f32')
    x_train /= 255
    x_val /= 255
    x_train, batch_count = dz.data.utils.dataset_from_ndarray_with_batch_count(x_train, 32)
    x_val, _ = dz.data.utils.dataset_from_ndarray_with_batch_count(x_val, 32)
    model.train(x_train, x_val, epochs=1, batch_count=batch_count, save_path='tests/saves')

def test_model_with_tf_dataset_no_batch_count(capsys):
    model = dz.Model(dz.nets.encoders.Encoder_32x32(), dz.nets.decoders.Decoder_32x32())
    x_train, x_val = dz.data.cifar10.load(64, 'f32')
    x_train /= 255
    x_val /= 255
    x_train, _ = dz.data.utils.dataset_from_ndarray_with_batch_count(x_train, 32)
    x_val, _ = dz.data.utils.dataset_from_ndarray_with_batch_count(x_val, 32)
    model.train(x_train, x_val, epochs=1, verbosity=2, save_path='tests/saves')
    captured = capsys.readouterr()
    assert(dz.Model._verbosity_compatability_warning in captured.out)

def test_model_with_incorrect_dataset_dtypes(capsys):
    model = dz.Model(dz.nets.encoders.Encoder_32x32(), dz.nets.decoders.Decoder_32x32())
    with pytest.raises(ValueError):
        model.train([[1.,1.],[2.,2.]], [[3.]], epochs=1, verbosity=2, save_path='tests/saves')
        captured = capsys.readouterr()
        assert(dz.Model._dataset_consistency_warning(None, '', '')[:10] in captured.err)

def test_model_with_np_no_batch_size(capsys):
    model = dz.Model(dz.nets.encoders.Encoder_32x32(), dz.nets.decoders.Decoder_32x32())
    x_train, x_val = dz.data.cifar10.load(64, 'f32')
    x_train /= 255
    x_val /= 255
    with pytest.raises(ValueError):
        model.train(x_train, x_val, epochs=1, verbosity=2, save_path='tests/saves')
        captured = capsys.readouterr()
        assert(dz.Model._batch_size_warning in captured.err)

def test_model_with_np_extra_batch_count(capsys):
    model = dz.Model(dz.nets.encoders.Encoder_32x32(), dz.nets.decoders.Decoder_32x32())
    x_train, x_val = dz.data.cifar10.load(64, 'f32')
    x_train /= 255
    x_val /= 255
    model.train(x_train, x_val, epochs=1, batch_size=32, verbosity=2, batch_count=100000, save_path='tests/saves')
    captured = capsys.readouterr()
    assert(dz.Model._batch_count_warning in captured.out)

def test_model_with_tf_extra_batch_size(capsys):
    model = dz.Model(dz.nets.encoders.Encoder_32x32(), dz.nets.decoders.Decoder_32x32())
    x_train, x_val = dz.data.cifar10.load(64, 'f32')
    x_train /= 255
    x_val /= 255
    x_train, _ = dz.data.utils.np_convert_to_tf(x_train, 32)
    x_val, _ = dz.data.utils.np_convert_to_tf(x_val, 32)
    model.train(x_train, x_val, epochs=1, batch_size=32, verbosity=2, batch_count=100000, save_path='tests/saves')
    captured = capsys.readouterr()
    assert(dz.Model._tf_batch_size_warning in captured.out)










