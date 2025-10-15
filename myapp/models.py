from django.contrib.auth.models import AbstractUser
from django.db import models
from django.conf import settings

class CustomUser(AbstractUser):
    ROLE_CHOICES = (
        ('admin', 'Admin'),
        ('client', 'Client'),
    )

    email = models.EmailField(unique=True)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='client')
    is_verified = models.BooleanField(default=False)  # <--- Add this

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']


class Bird(models.Model):
    name = models.CharField(max_length=255)
    scientific_name = models.CharField(max_length=255)
    description = models.TextField()
    population = models.CharField(max_length=100)
    class_index = models.IntegerField(unique=True)  # Corresponds to the model's output class index
    image = models.TextField(null=True, blank=True)  # <-- Changed to TextField

    def __str__(self):
        return self.name

class BirdPhoto(models.Model):  # Move outside of CustomUser
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)  # Correct reference
    photo = models.ImageField(upload_to='bird_photos/')
    bird = models.ForeignKey(Bird, on_delete=models.SET_NULL, null=True, blank=True)  # <-- add this
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.uploaded_at}"
    

class Signatory(models.Model):
    prepared_by = models.CharField(max_length=100, default="Default Preparer")
    approved_by = models.CharField(max_length=100, default="Default Approver")
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.prepared_by} / {self.approved_by}"


