def generate_synthetic_code_data():
    """Generate more code patterns synthetically"""
    
    # PHP patterns
    php_patterns = []
    for i in range(50):
        php_patterns.append(f"""
    class Model{i} {{
        private $id;
        private $name;
        
        public function __construct($id, $name) {{
            $this->id = $id;
            $this->name = $name;
        }}
        
        public function getId() {{
            return $this->id;
        }}
    }}
    """)
    
    # React patterns
    react_patterns = []
    components = ['Button', 'Input', 'Card', 'Modal', 'List']
    for comp in components:
        react_patterns.append(f"""
    const {comp} = ({{ children, onClick }}) => {{
        return (
            <div className="{comp.lower()}" onClick={{onClick}}>
                {{children}}
            </div>
        );
    }};
    """)
    
    return "\n".join(php_patterns + react_patterns)

if __name__ == "__main__":
    synthetic_data = generate_synthetic_code_data()
    print(f"Generated {len(synthetic_data)} characters of synthetic code")
    
    with open('synthetic_code_data.txt', 'w') as f:
        f.write(synthetic_data)
    
    print("Saved to synthetic_code_data.txt")