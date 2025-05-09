from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class CustomUserCreationForm(UserCreationForm):
    username = forms.CharField(
        max_length=150,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Choose a username'})
    )
    password1 = forms.CharField(
        required=True,
        label='Password',
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Create a password'})
    )
    password2 = forms.CharField(
        required=True,
        label='Confirm Password',
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Confirm your password'})
    )

    class Meta:
        model = User
        fields = ['username', 'password1', 'password2']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.help_text = None

class PredictForm(forms.Form):
    HomePlanet = forms.ChoiceField(
        choices=[
            ('Earth', 'Earth'),
            ('Mars', 'Mars'),
            ('Europa', 'Europa')
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    CryoSleep = forms.ChoiceField(
        choices=[
            ('True', 'Yes'),
            ('False', 'No')
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    Destination = forms.ChoiceField(
        choices=[
            ('TRAPPIST-1e', 'TRAPPIST-1e'),
            ('PSO J318.5-22', 'PSO J318.5-22'),
            ('55 Cancri e', '55 Cancri e')
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    Age = forms.FloatField(
        min_value=0,
        max_value=100,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter age (0-100)'})
    )
    
    VIP = forms.ChoiceField(
        choices=[
            ('True', 'Yes'),
            ('False', 'No')
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    RoomService = forms.FloatField(
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter room service amount'})
    )
    
    FoodCourt = forms.FloatField(
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter food court amount'})
    )
    
    ShoppingMall = forms.FloatField(
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter shopping mall amount'})
    )
    
    Spa = forms.FloatField(
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter spa amount'})
    )
    
    VRDeck = forms.FloatField(
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter VR deck amount'})
    )

    def clean(self):
        cleaned_data = super().clean()
        # Add any additional validation if needed
        return cleaned_data
