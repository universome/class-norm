firelab start configs/zsl.yml --config.silent True --config.dataset $1 --config.hp.model.type linear --config.hp.model.scale 1 --config.hp.model.has_bn false --config.hp.model.init.type kaiming
firelab start configs/zsl.yml --config.silent True --config.dataset $1 --config.hp.model.type linear --config.hp.model.scale 3 --config.hp.model.has_bn false --config.hp.model.init.type kaiming
firelab start configs/zsl.yml --config.silent True --config.dataset $1 --config.hp.model.type linear --config.hp.model.scale 5 --config.hp.model.has_bn false --config.hp.model.init.type kaiming
firelab start configs/zsl.yml --config.silent True --config.dataset $1 --config.hp.model.type linear --config.hp.model.scale 10 --config.hp.model.has_bn false --config.hp.model.init.type kaiming
firelab start configs/zsl.yml --config.silent True --config.dataset $1 --config.hp.model.type linear --config.hp.model.scale 20 --config.hp.model.has_bn false --config.hp.model.init.type kaiming

firelab start configs/zsl.yml --config.silent True --config.dataset $1 --config.hp.model.num_additional_hidden_layers 0 --config.hp.model.scale 1 --config.hp.model.has_bn false --config.hp.model.init.type kaiming
firelab start configs/zsl.yml --config.silent True --config.dataset $1 --config.hp.model.num_additional_hidden_layers 0 --config.hp.model.scale 3 --config.hp.model.has_bn false --config.hp.model.init.type kaiming
firelab start configs/zsl.yml --config.silent True --config.dataset $1 --config.hp.model.num_additional_hidden_layers 0 --config.hp.model.scale 5 --config.hp.model.has_bn false --config.hp.model.init.type kaiming
firelab start configs/zsl.yml --config.silent True --config.dataset $1 --config.hp.model.num_additional_hidden_layers 0 --config.hp.model.scale 10 --config.hp.model.has_bn false --config.hp.model.init.type kaiming
firelab start configs/zsl.yml --config.silent True --config.dataset $1 --config.hp.model.num_additional_hidden_layers 0 --config.hp.model.scale 20 --config.hp.model.has_bn false --config.hp.model.init.type kaiming

firelab start configs/zsl.yml --config.silent True --config.dataset $1 --config.hp.model.scale 1 --config.hp.model.has_bn false --config.hp.model.init.type kaiming
firelab start configs/zsl.yml --config.silent True --config.dataset $1 --config.hp.model.scale 3 --config.hp.model.has_bn false --config.hp.model.init.type kaiming
firelab start configs/zsl.yml --config.silent True --config.dataset $1 --config.hp.model.scale 5 --config.hp.model.has_bn false --config.hp.model.init.type kaiming
firelab start configs/zsl.yml --config.silent True --config.dataset $1 --config.hp.model.scale 10 --config.hp.model.has_bn false --config.hp.model.init.type kaiming
firelab start configs/zsl.yml --config.silent True --config.dataset $1 --config.hp.model.scale 20 --config.hp.model.has_bn false --config.hp.model.init.type kaiming

firelab start configs/zsl.yml --config.silent True --config.dataset $1 --config.hp.model.has_bn false --config.hp.model.has_dn true --config.hp.model.init.type kaiming

firelab start configs/zsl.yml --config.silent True --config.dataset $1 --config.hp.model.has_bn true --config.hp.model.init.type kaiming --config.hp.model.init.mode fan_in
firelab start configs/zsl.yml --config.silent True --config.dataset $1 --config.hp.model.has_bn true --config.hp.model.init.type kaiming --config.hp.model.init.mode fan_out
firelab start configs/zsl.yml --config.silent True --config.dataset $1 --config.hp.model.has_bn true --config.hp.model.init.type xavier
