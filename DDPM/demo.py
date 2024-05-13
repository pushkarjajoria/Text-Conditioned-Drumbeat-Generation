from main_latent_space import demo
import warnings

# Filter out the specific warnings
warnings.filterwarnings("ignore", message="`resume_download` is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message="enable_nested_tensor is True", category=UserWarning)

if __name__ == '__main__':
    demo()