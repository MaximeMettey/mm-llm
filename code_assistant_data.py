def get_php_data():
    """PHP code examples and patterns"""
    return """
    <?php
    class User {
        private $id;
        private $name;
        
        public function __construct($id, $name) {
            $this->id = $id;
            $this->name = $name;
        }
        
        public function getName() {
            return $this->name;
        }
    }
    
    $user = new User(1, "John");
    echo $user->getName();
    
    function calculateTotal($items) {
        $total = 0;
        foreach ($items as $item) {
            $total += $item['price'];
        }
        return $total;
    }
    
    $pdo = new PDO('mysql:host=localhost;dbname=app', $user, $pass);
    $stmt = $pdo->prepare("SELECT * FROM users WHERE id = ?");
    $stmt->execute([$id]);
    $user = $stmt->fetch();
    
    if ($user) {
        echo "User found: " . $user['name'];
    } else {
        echo "User not found";
    }
    ?>
    """

def get_react_data():
    """React and React Native examples"""
    return """
    import React, { useState, useEffect } from 'react';
    
    function UserProfile({ userId }) {
        const [user, setUser] = useState(null);
        const [loading, setLoading] = useState(true);
        
        useEffect(() => {
            fetchUser(userId).then(userData => {
                setUser(userData);
                setLoading(false);
            });
        }, [userId]);
        
        if (loading) return <div>Loading...</div>;
        
        return (
            <div className="user-profile">
                <h2>{user.name}</h2>
                <p>{user.email}</p>
            </div>
        );
    }
    
    const TodoList = () => {
        const [todos, setTodos] = useState([]);
        const [newTodo, setNewTodo] = useState('');
        
        const addTodo = () => {
            setTodos([...todos, { id: Date.now(), text: newTodo, done: false }]);
            setNewTodo('');
        };
        
        return (
            <div>
                <input 
                    value={newTodo}
                    onChange={(e) => setNewTodo(e.target.value)}
                    placeholder="Add todo..."
                />
                <button onClick={addTodo}>Add</button>
                {todos.map(todo => (
                    <div key={todo.id}>{todo.text}</div>
                ))}
            </div>
        );
    };
    
    // React Native example
    import { View, Text, TouchableOpacity } from 'react-native';
    
    const Button = ({ onPress, title }) => (
        <TouchableOpacity onPress={onPress}>
            <Text>{title}</Text>
        </TouchableOpacity>
    );
    """

def get_python_data():
    """Python code examples"""
    return """
    class UserManager:
        def __init__(self):
            self.users = []
            
        def add_user(self, name, email):
            user = {
                'id': len(self.users) + 1,
                'name': name,
                'email': email
            }
            self.users.append(user)
            return user
            
        def get_user(self, user_id):
            return next((u for u in self.users if u['id'] == user_id), None)
    
    def fetch_data(url):
        import requests
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None
    
    # List comprehension
    numbers = [1, 2, 3, 4, 5]
    squared = [x**2 for x in numbers if x % 2 == 0]
    
    # Decorator example
    def timer(func):
        import time
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"{func.__name__} took {end - start:.2f} seconds")
            return result
        return wrapper
    
    @timer
    def slow_function():
        time.sleep(1)
        return "Done"
    """

def get_symfony_data():
    """Symfony framework examples"""
    return """
    <?php
    namespace App\\Controller;
    
    use Symfony\\Bundle\\FrameworkBundle\\Controller\\AbstractController;
    use Symfony\\Component\\HttpFoundation\\JsonResponse;
    use Symfony\\Component\\Routing\\Annotation\\Route;
    use App\\Entity\\User;
    use App\\Repository\\UserRepository;
    
    class UserController extends AbstractController
    {
        #[Route('/api/users', methods: ['GET'])]
        public function getUsers(UserRepository $userRepository): JsonResponse
        {
            $users = $userRepository->findAll();
            return $this->json($users);
        }
        
        #[Route('/api/users/{id}', methods: ['GET'])]
        public function getUser(int $id, UserRepository $userRepository): JsonResponse
        {
            $user = $userRepository->find($id);
            
            if (!$user) {
                return $this->json(['error' => 'User not found'], 404);
            }
            
            return $this->json($user);
        }
        
        #[Route('/api/users', methods: ['POST'])]
        public function createUser(Request $request): JsonResponse
        {
            $data = json_decode($request->getContent(), true);
            
            $user = new User();
            $user->setName($data['name']);
            $user->setEmail($data['email']);
            
            $this->entityManager->persist($user);
            $this->entityManager->flush();
            
            return $this->json($user, 201);
        }
    }
    ?>
    """

def get_api_platform_data():
    """API Platform examples"""
    return """
    <?php
    namespace App\\Entity;
    
    use ApiPlatform\\Core\\Annotation\\ApiResource;
    use Doctrine\\ORM\\Mapping as ORM;
    use Symfony\\Component\\Serializer\\Annotation\\Groups;
    
    #[ApiResource(
        collectionOperations: [
            'get' => ['normalization_context' => ['groups' => 'user:read']],
            'post' => ['denormalization_context' => ['groups' => 'user:write']]
        ],
        itemOperations: [
            'get' => ['normalization_context' => ['groups' => 'user:read']],
            'put' => ['denormalization_context' => ['groups' => 'user:write']],
            'delete'
        ]
    )]
    #[ORM\\Entity]
    class User
    {
        #[ORM\\Id]
        #[ORM\\GeneratedValue]
        #[ORM\\Column(type: 'integer')]
        #[Groups(['user:read'])]
        private $id;
        
        #[ORM\\Column(type: 'string', length: 255)]
        #[Groups(['user:read', 'user:write'])]
        private $name;
        
        #[ORM\\Column(type: 'string', length: 255)]
        #[Groups(['user:read', 'user:write'])]
        private $email;
        
        public function getId(): ?int
        {
            return $this->id;
        }
        
        public function getName(): ?string
        {
            return $this->name;
        }
        
        public function setName(string $name): self
        {
            $this->name = $name;
            return $this;
        }
    }
    ?>
    """

def get_javascript_data():
    """Vanilla JavaScript examples"""
    return """
    // DOM manipulation
    const button = document.getElementById('myButton');
    button.addEventListener('click', function() {
        console.log('Button clicked!');
    });
    
    // Fetch API
    async function fetchUsers() {
        try {
            const response = await fetch('/api/users');
            const users = await response.json();
            return users;
        } catch (error) {
            console.error('Error fetching users:', error);
            return [];
        }
    }
    
    // Class example
    class TodoApp {
        constructor() {
            this.todos = [];
            this.init();
        }
        
        init() {
            this.render();
            this.bindEvents();
        }
        
        addTodo(text) {
            const todo = {
                id: Date.now(),
                text: text,
                completed: false
            };
            this.todos.push(todo);
            this.render();
        }
        
        toggleTodo(id) {
            const todo = this.todos.find(t => t.id === id);
            if (todo) {
                todo.completed = !todo.completed;
                this.render();
            }
        }
        
        render() {
            const container = document.getElementById('todos');
            container.innerHTML = this.todos.map(todo => 
                `<div class="${todo.completed ? 'completed' : ''}">
                    ${todo.text}
                </div>`
            ).join('');
        }
    }
    
    // Event handling
    document.addEventListener('DOMContentLoaded', function() {
        const app = new TodoApp();
    });
    """

def get_conversational_data():
    """Conversational patterns for code assistance"""
    return """
    How do I create a PHP class? You can create a PHP class using the class keyword followed by the class name.
    
    Can you show me a React component? Sure! Here's a simple React functional component that uses hooks.
    
    What's the syntax for a Python function? Python functions are defined using the def keyword followed by the function name.
    
    How do I make an API call in JavaScript? You can use the fetch API or axios to make HTTP requests.
    
    Can you help me with Symfony routing? Symfony uses annotations or attributes to define routes in controllers.
    
    What is API Platform? API Platform is a framework for building REST and GraphQL APIs in PHP.
    
    How do I handle forms in React? You can use controlled components with useState to handle form inputs.
    
    Show me error handling in PHP. You can use try-catch blocks or set error handlers in PHP.
    
    What's the difference between let and const? let allows reassignment while const creates immutable bindings.
    
    Can you explain React hooks? Hooks are functions that let you use state and lifecycle features in functional components.
    """

def get_combined_code_dataset():
    """Combine all code examples with conversational elements"""
    return (
        get_conversational_data() + "\n" +
        get_php_data() + "\n" +
        get_react_data() + "\n" +
        get_python_data() + "\n" +
        get_symfony_data() + "\n" +
        get_api_platform_data() + "\n" +
        get_javascript_data()
    )

if __name__ == "__main__":
    dataset = get_combined_code_dataset()
    print(f"Combined dataset length: {len(dataset)} characters")
    
    with open('code_assistant_data.txt', 'w') as f:
        f.write(dataset)
    
    print("Code assistant dataset saved to code_assistant_data.txt")