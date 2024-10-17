6. **Project Structure**

```
project-root/
  |-- pom.xml
  |-- users-module/
       |-- pom.xml
       |-- src/main/java
       |     |-- com.example.users
       |          |-- User.java
       |          |-- UserController.java
       |          |-- UserService.java
       |          |-- UserRepository.java
  |-- sellers-module/
       |-- pom.xml
       |-- src/main/java
       |     |-- com.example.sellers
       |          |-- SellerDetails.java
       |          |-- SellerController.java
       |          |-- SellerService.java
       |          |-- SellerRepository.java
  |-- products-module/
       |-- pom.xml
       |-- src/main/java
       |     |-- com.example.products
       |          |-- Product.java
       |          |-- ProductController.java
       |          |-- ProductService.java
       |          |-- ProductRepository.java
  |-- orders-module/
       |-- pom.xml
       |-- src/main/java
       |     |-- com.example.orders
       |          |-- Order.java
       |          |-- OrderController.java
       |          |-- OrderService.java
       |          |-- OrderRepository.java
  |-- shipments-module/
       |-- pom.xml
       |-- src/main/java
       |     |-- com.example.shipments
       |          |-- Shipment.java
       |          |-- ShipmentController.java
       |          |-- ShipmentService.java
       |          |-- ShipmentRepository.java
  |-- auth-module/
       |-- pom.xml
       |-- src/main/java
       |     |-- com.example.auth
       |          |-- AuthController.java
       |          |-- AuthService.java
       |          |-- AuthRequest.java
  |-- diary-module/
       |-- pom.xml
       |-- src/main/java
       |     |-- com.example.diary
       |          |-- DiaryEntry.java
       |          |-- DiaryController.java
       |          |-- DiaryService.java
       |          |-- DiaryRepository.java
  |-- cart-module/
       |-- pom.xml
       |-- src/main/java
       |     |-- com.example.cart
       |          |-- CartItem.java
       |          |-- CartController.java
       |          |-- CartService.java
       |          |-- CartRepository.java
```

1. **Modify User Entity to Include UserType**

The `User` entity already contains the `userType` field, which is an enum that can be either `GENERAL` or `SELLER`.

```java
package com.example.users;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "Users")
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int userId;

    @Column(nullable = false)
    private String password;

    @Column(nullable = false, unique = true)
    private String email;

    @Column(nullable = false)
    private String name;

    private String phone;
    private String address;

    @Enumerated(EnumType.STRING)
    private UserType userType;
    private String account;

    @Column(nullable = false, updatable = false)
    private LocalDateTime createdAt = LocalDateTime.now();

    // Getters and Setters
}
```

2. **Update Authentication Logic to Differentiate Users**

In the `AuthService`, add logic to return different responses or handle login differently depending on whether the user is a `GENERAL` user or a `SELLER`.

**AuthService.java**

```java
package com.example.auth;

import com.example.users.User;
import com.example.users.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.bcrypt.BCrypt;
import org.springframework.stereotype.Service;

@Service
public class AuthService {

    @Autowired
    private UserRepository userRepository;

    public User signup(User user) {
        user.setPassword(BCrypt.hashpw(user.getPassword(), BCrypt.gensalt()));
        return userRepository.save(user);
    }

    public User login(String email, String password) {
        User user = userRepository.findByEmail(email);
        if (user != null && BCrypt.checkpw(password, user.getPassword())) {
            return user;
        }
        return null;
    }
}
```

3. **Add UserType-Based Endpoint Handling in Controllers**

Modify the `AuthController` to return different responses based on the `userType` of the logged-in user.

**AuthController.java**

```java
package com.example.auth;

import com.example.users.User;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/auth")
public class AuthController {

    @Autowired
    private AuthService authService;

    @PostMapping("/signup")
    public ResponseEntity<User> signup(@RequestBody User user) {
        User createdUser = authService.signup(user);
        return ResponseEntity.ok(createdUser);
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody AuthRequest authRequest) {
        User user = authService.login(authRequest.getEmail(), authRequest.getPassword());
        if (user == null) {
            return ResponseEntity.status(401).build();
        }
        
        if (user.getUserType() == UserType.SELLER) {
            return ResponseEntity.ok("SELLER");
        } else {
            return ResponseEntity.ok("GENERAL");
        }
    }
}
```

4. **Create Seller-Specific Features - My Shop**

Add a `MyShopController` for sellers to manage their shop.

**MyShopController.java**

```java
package com.example.sellers;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/myshop")
public class MyShopController {

    @Autowired
    private SellerService sellerService;

    @GetMapping("/{sellerId}/products")
    public List<Product> getProductsBySeller(@PathVariable int sellerId) {
        return sellerService.getProductsBySeller(sellerId);
    }

    @PostMapping("/{sellerId}/product")
    public Product addProduct(@PathVariable int sellerId, @RequestBody Product product) {
        return sellerService.addProduct(sellerId, product);
    }
}
```

**SellerService.java**

```java
package com.example.sellers;

import com.example.products.Product;
import com.example.products.ProductRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class SellerService {

    @Autowired
    private ProductRepository productRepository;

    public List<Product> getProductsBySeller(int sellerId) {
        return productRepository.findBySellerId(sellerId);
    }

    public Product addProduct(int sellerId, Product product) {
        product.setSellerId(sellerId);
        return productRepository.save(product);
    }
}
```

5. **Create User-Specific Features - My Page**

Add a `MyPageController` for general users to manage their information and view orders.

**MyPageController.java**

```java
package com.example.users;

import com.example.orders.Order;
import com.example.orders.OrderService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/mypage")
public class MyPageController {

    @Autowired
    private UserService userService;

    @Autowired
    private OrderService orderService;

    @GetMapping("/{userId}")
    public User getUserDetails(@PathVariable int userId) {
        return userService.getUserById(userId);
    }

    @GetMapping("/{userId}/orders")
    public List<Order> getUserOrders(@PathVariable int userId) {
        return orderService.getOrdersByUserId(userId);
    }
}
```
