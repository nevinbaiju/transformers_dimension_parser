import random
from config import vec_len, data_templates

def generate_synthetic_data(templates):
    template = random.choice(templates)
    height = random.randint(10, 99)
    width = random.randint(10, 99)
    synthetic_data = template.format(height, width)
    return f'{synthetic_data},{height},{width}\n'

if __name__=="__main__":
    with open('data.csv', 'w') as file:
        file.write('description,height,width\n')
        for i in range(1000000):
            file.write(generate_synthetic_data(data_templates))