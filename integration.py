import torch
import torch.nn as nn
import argparse
from models import Teacher, Student, Generator  # Import from your existing DAFL code
from resource_aware_dafl import ResourceAwareDAFL
#here argspace is used for parser 
def main():
    parser = argparse.ArgumentParser(description='Resource-Aware DAFL Training')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'mnist'],
                        help='dataset to use')
    parser.add_argument('--teacher-path', type=str, required=True, help='path to teacher model')
    parser.add_argument('--resource-mode', type=str, default='both', 
                        choices=['standard', 'layerwise', 'split', 'both'],
                        help='resource-saving approach to use')
    parser.add_argument('--split-layer', type=int, default=3, 
                        help='layer at which to split the model for split learning')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Initialize models (adapt this part based on your existing code)
    if args.dataset == 'cifar10':
        teacher_model = Teacher(num_classes=10)
        student_model = Student(num_classes=10)
        generator_model = Generator(nz=args.nz)
    elif args.dataset == 'cifar100':
        teacher_model = Teacher(num_classes=100)
        student_model = Student(num_classes=100)
        generator_model = Generator(nz=args.nz)
    else:  # mnist
        teacher_model = Teacher(num_classes=10, input_channels=1)
        student_model = Student(num_classes=10, input_channels=1)
        generator_model = Generator(nz=args.nz, output_channels=1)
    
    # Load teacher model
    teacher_model.load_state_dict(torch.load(args.teacher_path))
    
    # Determine which resource-saving approaches to use
    use_layerwise = args.resource_mode in ['layerwise', 'both']
    use_split_learning = args.resource_mode in ['split', 'both']
    
    # Initialize the resource-aware DAFL trainer
    dafl_trainer = ResourceAwareDAFL(
        teacher_model=teacher_model,
        student_model=student_model,
        generator_model=generator_model,
        device=device,
        use_layerwise=use_layerwise,
        use_split_learning=use_split_learning,
        split_layer=args.split_layer
    )
    
    # Train the student model with resource constraints
    dafl_trainer.train_student_with_resource_constraints(
        num_batches=500,
        batch_size=args.batch_size,
        nz=args.nz
    )
    
    # Save the trained student model
    torch.save(student_model.state_dict(), f'student_{args.dataset}_{args.resource_mode}.pth')
    print(f"Trained student model saved to student_{args.dataset}_{args.resource_mode}.pth")

if __name__ == '__main__':
    main()
