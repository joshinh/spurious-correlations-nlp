import os
import numpy as np
import torch
from datasets import concatenate_datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm


def get_cached_dataset(task_name: str, set_type, output_dir, verbose=True) -> torch.utils.data.Dataset:
    assert set_type in {'train', 'val', 'test'}
    file_name = f'{task_name}_{set_type}_cached.pkl'
    output_file = os.path.abspath(os.path.join(output_dir, file_name))
    if os.path.isfile(output_file):
        if verbose:
            print(f'Found cached dataset in {output_file}')
        data = torch.load(output_file)
        all_guid = data['guid']
        all_embeddings = data['examples']
        all_labels = data['labels']
        return TensorDataset(all_guid, all_embeddings, all_labels)
    raise IOError(f'Database not found at {output_file}')


def embed_and_cache(cache_tag: str, dataset, set_type, encoder, output_dir, balance=True, collate_fn=None, num_labels=2,
                    augment_z=False, save_groups=False, eval_groups=False, batch_size=32, device=None, overwrite_cache=False):
    assert set_type in {'train', 'val', 'test'}
    file_name = f'{cache_tag}_{set_type}_cached.pkl'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.abspath(os.path.join(output_dir, file_name))
    if overwrite_cache:
        print(f'Overwriting cached files...')
    elif os.path.isfile(output_file):
        print(f'Found cached dataset in {output_file}')
        data = torch.load(output_file)
        all_guid, all_embeddings, all_labels = data['guid'], data['examples'], data['labels']
        return TensorDataset(all_guid, all_embeddings, all_labels)
    print(f'Caching into {output_file}')
    

    all_labels = torch.tensor(dataset['probing_label'])
    # Only supports binary classification (currently)
    if num_labels == 2:
        num_positive_samples = (all_labels == 1).sum().item()
        print('total samples: ', len(all_labels))
        print('positive samples: ', num_positive_samples)
    elif num_labels == 3:
        print("WARNING: Number of labels in the probe is 3")
    else:
        raise NotImplementedError

    if balance and num_labels == 2:
        
        #### Default balancing
        ds_seed = 42
        positive_dataset = dataset.filter(lambda x: x['probing_label'] == 1, load_from_cache_file=not overwrite_cache)
        negative_dataset = dataset.filter(lambda x: x['probing_label'] == 0, load_from_cache_file=not overwrite_cache) \
          .shuffle(seed=ds_seed, load_from_cache_file=not overwrite_cache) \
          .select(np.arange(positive_dataset.num_rows))
        
        #### For synthetic bias, randomly sample 5k positive and negative examples
        #### (otherwise probing dataset is too big)
        
        # if set_type == "train":
        #     ds_seed = 42
        #     positive_dataset = dataset.filter(lambda x: x['probing_label'] == 1, load_from_cache_file=not overwrite_cache) \
        #        .shuffle(seed=ds_seed, load_from_cache_file=not overwrite_cache) \
        #        .select(np.arange(5000))
        #     negative_dataset = dataset.filter(lambda x: x['probing_label'] == 0, load_from_cache_file=not overwrite_cache) \
        #        .shuffle(seed=ds_seed, load_from_cache_file=not overwrite_cache) \
        #        .select(np.arange(5000))
        # else:
        #     ds_seed = 42
        #     positive_dataset = dataset.filter(lambda x: x['probing_label'] == 1, load_from_cache_file=not overwrite_cache)
        #     negative_dataset = dataset.filter(lambda x: x['probing_label'] == 0, load_from_cache_file=not overwrite_cache) \
        #       .shuffle(seed=ds_seed, load_from_cache_file=not overwrite_cache) \
        #       .select(np.arange(positive_dataset.num_rows))
        
        
        dataset = concatenate_datasets([positive_dataset, negative_dataset])\
            .shuffle(seed=ds_seed, load_from_cache_file=not overwrite_cache)
        num_samples = dataset.num_rows
    else:
        num_samples = len(all_labels)
        # sampler = None
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    all_guid = None
    all_embeddings = None
    all_labels = None
    if augment_z or save_groups:
        all_nuisance = None
    elif eval_groups:
        all_nli_labels = None
    if num_samples > 0:
        encoder = encoder.to(device)
        t = tqdm(dataloader, desc='Embedding')
        with torch.no_grad():
            for batch in t:
                guid = batch[0]
                inputs = batch[1]
                label = batch[2]
                
                if augment_z or save_groups:
                    nuisance = batch[3]
                elif eval_groups:
                    nli_label = batch[3]
                    
                
                ### RoBERTa
                embedding, _ = encoder(**inputs, return_dict=False)
                embedding = embedding[:,0]
                
                if all_guid is None:
                    all_guid = guid.numpy()
                    all_embeddings = embedding.cpu().numpy()
                    all_labels = label.numpy()
                    if augment_z or save_groups:
                        all_nuisance = nuisance.numpy()
                    elif eval_groups:
                        all_nli_labels = nli_label.numpy()
                else:
                    all_guid = np.append(all_guid, guid, axis=0)
                    all_embeddings = np.append(all_embeddings, embedding.cpu().numpy(), axis=0)
                    all_labels = np.append(all_labels, label.numpy(), axis=0)
                    if augment_z or save_groups:
                        all_nuisance = np.append(all_nuisance, nuisance.numpy(), axis=0)
                    elif eval_groups:
                        all_nli_labels = np.append(all_nli_labels, nli_label.numpy(), axis=0)

        all_guid = torch.tensor(all_guid)
        all_embeddings = torch.tensor(all_embeddings)
        all_labels = torch.tensor(all_labels)
        if augment_z:
            all_nuisance = torch.tensor(all_nuisance)[:, None]
            all_embeddings = torch.cat((all_embeddings, all_nuisance), dim=1)
        elif eval_groups:
            all_nli_labels = torch.tensor(all_nli_labels)
    else:
        all_guid = all_embeddings = all_labels = torch.tensor([])

    
    if save_groups:
        
        all_nuisance = torch.tensor(all_nuisance)
        
        label_list = [0,1,2]
        nuisance_list = [0,1]
        
        for x in label_list:
            for y in nuisance_list:
                idx = torch.nonzero(torch.logical_and((all_labels == x), (all_nuisance == y)), as_tuple=True)[0]
                file_name = f'{cache_tag}_{set_type}_label' + str(x) + '_nuisance' + str(y) + '.pkl'
                group_file = os.path.abspath(os.path.join(output_dir, file_name))
                torch.save({
                    'guid': all_guid[idx],
                    'examples': all_embeddings[idx],
                    'labels': all_labels[idx]
                }, group_file)
                
    
    if eval_groups:
        
        torch.save({
            'guid': all_guid,
            'examples': all_embeddings,
            'labels': all_labels,
            'nli_labels': all_nli_labels
        }, output_file)
        dataset = TensorDataset(all_guid, all_embeddings, all_labels, all_nli_labels)
        print('new dataset size: ', len(dataset))
        return dataset
        
    else:
    
        torch.save({
            'guid': all_guid,
            'examples': all_embeddings,
            'labels': all_labels
        }, output_file)
        dataset = TensorDataset(all_guid, all_embeddings, all_labels)
        print('new dataset size: ', len(dataset))
        return dataset
