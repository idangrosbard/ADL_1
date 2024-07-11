from dataset import setup_dataloaders

if __name__ == '__main__':
    train, test = setup_dataloaders(2, 'lra_clf')
    print(len(train), len(test))
    for t in train:
        print(t)
        break