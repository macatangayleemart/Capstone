from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from .forms import RegisterForm
from .models import CustomUser
from django.http import JsonResponse
import base64
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
import numpy as np
import tensorflow as tf
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import io
from .models import Bird, BirdPhoto
import json
from django.core.files.uploadedfile import InMemoryUploadedFile
import os
from django.conf import settings
from django.contrib.auth.decorators import user_passes_test
from django.http import HttpResponse
import base64
from django.http import Http404
from django.views.decorators.http import require_GET
from django.db.models import Count
from django.utils.timezone import now
from collections import defaultdict
from django.views.decorators.http import require_POST
from django.contrib.auth import get_user_model
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
import calendar
from django.db.models.functions import TruncMonth
from django.utils import timezone
from django.http import HttpResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from .models import BirdPhoto, Bird, CustomUser
from collections import Counter, defaultdict
from datetime import datetime
import io
from django.http import HttpResponse
from .models import BirdPhoto, Bird, CustomUser
from collections import Counter, defaultdict
from datetime import datetime
import csv
from django.contrib.auth.tokens import default_token_generator
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.template.loader import render_to_string
from django.core.mail import send_mail
from django.conf import settings
from django.urls import reverse
from django.http import StreamingHttpResponse
import cv2
import torch
from ultralytics import YOLO
import uuid
from .models import Signatory
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors    
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle



def register_view(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.role = "client"
            user.is_active = False  # Prevent login
            user.save()

            # Send verification email
            token = default_token_generator.make_token(user)
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            verify_url = request.build_absolute_uri(
                reverse("verify_email", kwargs={"uidb64": uid, "token": token})
            )
            print("Preparing to send verification email...")  # <--- DEBUG LINE

            subject = "Verify your email"
            message = f"Click the link to verify your email: {verify_url}"
            from_email = settings.DEFAULT_FROM_EMAIL
            to_email = [user.email]

            send_mail(subject, message, from_email, to_email)

            messages.success(request, "Check your email to verify your account.")
            return redirect('login')
    else:
        form = RegisterForm()

    return render(request, "register.html", {"form": form})






def login_view(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")

        user = authenticate(request, email=email, password=password)

        if user is not None:
            if not user.is_verified:
                messages.error(request, "Email not verified. Check your inbox.")
                return redirect('login')

            login(request, user)
            if user.role == "admin":
                return redirect("admin_dashboard")
            else:
                return redirect("user_dashboard")
        else:
            messages.error(request, "Invalid email or password.")

    return render(request, "login.html")
def verify_email(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = CustomUser.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, CustomUser.DoesNotExist):
        user = None

    if user and default_token_generator.check_token(user, token):
        user.is_active = True
        user.is_verified = True
        user.save()
        messages.success(request, "Email verified. You can now log in.")
        return redirect('login')
    else:
        messages.error(request, "Verification link is invalid or expired.")
        return redirect('login')


def logout_view(request):
    logout(request)
    return redirect('login')



@login_required 
def dashboard_view(request):
    if request.user.role == "admin":
        return render(request, 'admin_dashboard.html')  # Admin dashboard
    else:
        return render(request, 'user_dashboard.html')  # User dashboard

def home_view(request):
    return render(request, "home.html")



#Client Codessss//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
@csrf_exempt
def upload_photo(request):
        if request.method == "POST":
            image_data = request.POST.get("image")
            if image_data:
                try:
                    format, imgstr = image_data.split(";base64,")
                    ext = format.split("/")[-1]
                    image_file = ContentFile(base64.b64decode(imgstr), name=f"user_{request.user.id}.{ext}")

                    # Save the image manually first
                    file_path = os.path.join(settings.MEDIA_ROOT, "bird_photos", f"user_{request.user.id}.{ext}")
                    with open(file_path, "wb") as f:
                        f.write(base64.b64decode(imgstr))

                    # Now create a BirdPhoto entry
                    photo = BirdPhoto.objects.create(user=request.user, photo=f"bird_photos/user_{request.user.id}.{ext}")

                    return JsonResponse({"message": "Photo uploaded successfully!", "image_url": photo.photo.url})
                except Exception as e:
                    return JsonResponse({"error": str(e)}, status=500)

        return JsonResponse({"error": "Invalid request"}, status=400)


@login_required
def upload_history(request):
    user_photos = BirdPhoto.objects.filter(user=request.user).order_by('-uploaded_at')  # Fetch user uploads
    return render(request, 'upload_history.html', {'user_photos': user_photos})
    






def detector_view(request):
    return render(request, 'detector.html')

@login_required
def client_dashboard(request):
    total_uploads = BirdPhoto.objects.filter(user=request.user).count()

    return render(request, 'user_dashboard.html', {
        'total_uploads': total_uploads,
    })

@login_required
def user_upload_count_api(request):
    count = BirdPhoto.objects.filter(user=request.user).count()
    return JsonResponse({'total_uploads': count})

def user_detection_stats(request):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Unauthorized'}, status=401)

    bird_detections = (
        BirdPhoto.objects
        .filter(user=request.user, bird__isnull=False)
        .values('bird__name')
        .annotate(count=Count('id'))
        .order_by('-count')
    )

    labels = [item['bird__name'] for item in bird_detections]
    counts = [item['count'] for item in bird_detections]

    return JsonResponse({'labels': labels, 'counts': counts})

def user_dashboard(request):
    return render(request, 'user_dashboard.html')

def most_captured_bird_api(request):
    now = timezone.now()
    current_month_photos = BirdPhoto.objects.filter(
        uploaded_at__year=now.year,
        uploaded_at__month=now.month,
        bird__isnull=False
    )

    top_bird = (
        current_month_photos
        .values('bird__name')
        .annotate(count=Count('bird'))
        .order_by('-count')
        .first()
    )

    if top_bird:
        return JsonResponse({
            'bird_name': top_bird['bird__name'],
            'count': top_bird['count'],
            'month': now.strftime('%B')
        })
    else:
        return JsonResponse({
            'bird_name': 'No captures yet',
            'count': 0,
            'month': now.strftime('%B')
        })
    
def bird_count(request):
    count = Bird.objects.count()
    return JsonResponse({'count': count})

def client_bird_list(request):
    birds = Bird.objects.all()
    bird_list = []
    for bird in birds:
        if bird.image:
            # assuming your image is stored as base64 text in the database
            image_url = f"data:image/jpeg;base64,{bird.image}"
        else:
            image_url = ''
        bird_list.append({
            'name': bird.name,
            'image': image_url
        })
    return JsonResponse({'birds': bird_list})



# Build the absolute path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_waterbirds_model.pt")

# Load the YOLOv8 model once
model = YOLO(MODEL_PATH)
print(model.names)
print("Loaded from:", MODEL_PATH)
print("File exists?", os.path.exists(MODEL_PATH))

@csrf_exempt
def predict_bird(request):
    if request.method == "POST":
        frame_file = request.FILES.get("frame")
        if not frame_file:
            return JsonResponse({"error": "No frame received"}, status=400)

        # Convert to OpenCV format
        frame = frame_file.read()
        nparr = np.frombuffer(frame, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run YOLO detection
        results = model(img)
        detections = results[0].boxes

        birds_data = []
        if len(detections) > 0:
            for det in detections:
                cls_id = int(det.cls[0].item())
                bird_class = model.names[cls_id]
                conf = float(det.conf[0].item())
                x1, y1, x2, y2 = det.xyxy[0].tolist()

                try:
                    bird_obj = Bird.objects.get(name=bird_class)
                except Bird.DoesNotExist:
                    bird_obj = None

                # Save the uploaded frame into BirdPhoto
                # Save uploaded frame for each detected bird
            filename = f"{uuid.uuid4()}.jpg"
            for det in detections:
                cls_id = int(det.cls[0].item())
                bird_class = model.names[cls_id]
                conf = float(det.conf[0].item())
                x1, y1, x2, y2 = det.xyxy[0].tolist()

                try:
                    bird_obj = Bird.objects.get(name=bird_class)
                except Bird.DoesNotExist:
                    bird_obj = None

                # Each detection = one BirdPhoto entry
                bird_photo = BirdPhoto.objects.create(
                    user=request.user,
                    bird=bird_obj,
                )
                bird_photo.photo.save(filename, ContentFile(frame), save=True)

                birds_data.append({
                    "name": bird_obj.name if bird_obj else bird_class,
                    "scientific_name": bird_obj.scientific_name if bird_obj else "Unknown",
                    "population": bird_obj.population if bird_obj else "Unknown",
                    "description": bird_obj.description if bird_obj else "No description",
                    "image_url": bird_obj.image if bird_obj and bird_obj.image else None,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })


        return JsonResponse({"birds": birds_data})

    return JsonResponse({"error": "Invalid request"}, status=400)


@csrf_exempt
def get_last_detection(request):
    if request.method == "POST":
        # Example: save to BirdPhoto history
        # You may want to attach to the logged-in user
        BirdPhoto.objects.create(
            user=request.user if request.user.is_authenticated else None,
            bird_name="Detected Bird",
        )
        return JsonResponse({"success": True, "message": "Detection saved."})

    return JsonResponse({"success": False, "message": "Invalid request."})








#Admin codessss/////////////////////////////////////////////////////////////////////////////////////////////////////////
def admin_dashboard(request):
    total_birds = Bird.objects.count()
    User = get_user_model()
    total_users = User.objects.count()
    total_photos = BirdPhoto.objects.count()

    return render(request, 'admin_dashboard.html', {
        'total_birds': total_birds,
        'total_users': total_users,
        'total_photos': total_photos
    })

# Optional if you need AJAX for total users too
def get_total_users(request):
    User = get_user_model()
    total_users = User.objects.count()
    return JsonResponse({'total_users': total_users})


def get_total_birds(request):
    total_birds = Bird.objects.count()
    return JsonResponse({'total_birds': total_birds})

def get_total_photos(request):
    total_photos = BirdPhoto.objects.count()
    return JsonResponse({'total_photos': total_photos})

def is_admin(user):
    return hasattr(user, "role") and user.role.lower() == "admin"

@login_required
def user_uploads(request):
    if hasattr(request.user, "role") and request.user.role.lower() == "admin":
        return render(request, "user_uploads.html", {"photos": BirdPhoto.objects.all()})

    return redirect("user_dashboard")



def bird_detection_stats(request):
    bird_counts = (
        BirdPhoto.objects
        .filter(bird__isnull=False)
        .values('bird__name')
        .annotate(count=Count('id'))
        .order_by('-count')
    )

    data = {
        "labels": [item['bird__name'] for item in bird_counts],
        "counts": [item['count'] for item in bird_counts],
    }

    return JsonResponse(data)

@csrf_exempt  # Only use this for testing. Prefer CSRF token for security.
def update_bird(request):
    if request.method == 'POST':
        bird_id = request.POST.get('id')
        try:
            bird = Bird.objects.get(pk=bird_id)
            bird.name = request.POST.get('name')
            bird.scientific_name = request.POST.get('scientific_name')
            bird.description = request.POST.get('description')
            bird.population = request.POST.get('population')

            # Check if a new image file is uploaded
            if 'image' in request.FILES:
                bird.image = request.FILES['image']
            # If no new image uploaded, keep the old one (do nothing here)

            bird.save()
            return JsonResponse({'status': 'success'})
        except Bird.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Bird not found'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})

def birds_list(request):
    return render(request, 'birds_list.html')

@require_GET
def get_birds(request):
    try:
        birds = Bird.objects.all()

        birds_data = []
        for bird in birds:
            print(f"Processing bird: {bird.name}")

            image_data = bird.image if bird.image else ''

            birds_data.append({
                'id': bird.id,
                'name': bird.name,
                'scientific_name': bird.scientific_name,
                'description': bird.description,
                'population': bird.population,
                'image_data': image_data,
            })

        if not birds_data:
            print("No birds data found.")

        return JsonResponse({'birds': birds_data}, status=200)

    except Exception as e:
        print(f"Error in get_birds: {e}")
        return JsonResponse({'error': f'Something went wrong while fetching birds: {str(e)}'}, status=500)
    
@csrf_exempt
def add_bird(request):
    if request.method == 'POST':
        name = request.POST['name']
        scientific_name = request.POST['scientific_name']
        description = request.POST['description']
        population = request.POST['population']
        class_index = request.POST['class_index']
        image_file = request.FILES.get('image')

        if image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        else:
            image_data = None

        Bird.objects.create(
            name=name,
            scientific_name=scientific_name,
            description=description,
            population=population,
            class_index=class_index,
            image=image_data
        )
        # Redirect or return success
        # ✅ Send JSON success response
        return JsonResponse({'success': True})

    return JsonResponse({'error': 'Invalid request method.'}, status=400)

def bird_detection_per_bird_monthly(request):
    from collections import defaultdict

    # Step 1: Get all birds
    all_birds = Bird.objects.values_list('name', flat=True)

    # Step 2: Get detections by bird/month
    detections = (
        BirdPhoto.objects
        .filter(bird__isnull=False)
        .annotate(month=TruncMonth('uploaded_at'))
        .values('bird__name', 'month')
        .annotate(total=Count('id'))
        .order_by('month')
    )

    # Step 3: Extract unique months from detections
    months = sorted({d['month'].month for d in detections})
    month_labels = [calendar.month_abbr[m] for m in months]

    # Step 4: Initialize all birds with 0s for all months
    birds_data = {bird: {m: 0 for m in months} for bird in all_birds}

    for entry in detections:
        bird = entry['bird__name']
        month = entry['month'].month
        birds_data[bird][month] = entry['total']

    # Step 5: Prepare datasets
    colors = [
        'rgba(255, 99, 132, 1)', 'rgba(54, 162, 235, 1)', 'rgba(255, 206, 86, 1)',
        'rgba(75, 192, 192, 1)', 'rgba(153, 102, 255, 1)', 'rgba(255, 159, 64, 1)',
        'rgba(199, 199, 199, 1)', 'rgba(83, 102, 255, 1)', 'rgba(99, 255, 132, 1)'
    ]

    datasets = []
    for i, (bird, counts) in enumerate(birds_data.items()):
        datasets.append({
            'label': bird,
            'data': [counts[m] for m in months],
            'borderColor': colors[i % len(colors)],
            'backgroundColor': colors[i % len(colors)],
            'fill': False,
            'tension': 0.3,
            'pointRadius': 3
        })

    return JsonResponse({'labels': month_labels, 'datasets': datasets})

def reports_view(request):
    signatory, created = Signatory.objects.get_or_create(id=1)

    if request.method == "POST":
        signatory.prepared_by = request.POST.get("prepared_by")
        signatory.approved_by = request.POST.get("approved_by")
        signatory.save()
        return redirect("reports")

    return render(request, "reports.html", {"signatory": signatory})

# 1. Bird Detections Report
def export_bird_detections_csv(request):
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="bird_detections.csv"'

    writer = csv.writer(response)

    # Title and timestamp
    writer.writerow(['Bird Detections Report'])
    writer.writerow(['Exported on:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    writer.writerow([])

    # Table headers
    writer.writerow(['Bird Name', 'Detected By', 'Date Detected'])

    for photo in BirdPhoto.objects.select_related('bird', 'user'):
        bird_name = photo.bird.name if photo.bird else "Unknown"
        writer.writerow([
            bird_name,
            photo.user.username,
            photo.uploaded_at.strftime('%Y-%m-%d %H:%M:%S')
        ])
    return response


# 2. User Activity Report
def export_user_activity_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="user_activity.csv"'

    writer = csv.writer(response)

    # Title and timestamp
    writer.writerow(['User Activity Report'])
    writer.writerow(['Exported on:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    writer.writerow([])

    # Table headers
    writer.writerow(['Username', 'Email', 'Total Detections'])

    for user in CustomUser.objects.all():
        total_detections = BirdPhoto.objects.filter(user=user).count()
        writer.writerow([
            user.username,
            user.email,
            f"{total_detections:,}"  # Format number with commas
        ])
    return response


# 3. Monthly Trends Report (with most detected bird)
def export_monthly_trends_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="monthly_trends.csv"'

    writer = csv.writer(response)

    # Title and timestamp
    writer.writerow(['Monthly Bird Detection Trends'])
    writer.writerow(['Exported on:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    writer.writerow([])

    # Table headers
    writer.writerow(['Month', 'Total Detections', 'Most Detected Bird'])

    detections = BirdPhoto.objects.select_related('bird').all()
    monthly_data = defaultdict(list)

    for detection in detections:
        month_year = detection.uploaded_at.strftime('%B %Y')
        if detection.bird:
            monthly_data[month_year].append(detection.bird.name)

    for month, birds in sorted(monthly_data.items()):
        total = len(birds)
        most_common = Counter(birds).most_common(1)
        top_bird = most_common[0][0] if most_common else "N/A"
        writer.writerow([month, f"{total:,}", top_bird])
    return response


# 4. Complete Bird Database Report
def export_bird_database_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="bird_database.csv"'

    writer = csv.writer(response)

    # Title and timestamp
    writer.writerow(['Complete Bird Database Report'])
    writer.writerow(['Exported on:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    writer.writerow([])

    # Table headers
    writer.writerow(['Name', 'Scientific Name', 'Population', 'Description'])

    for bird in Bird.objects.all():
        writer.writerow([
            bird.name,
            bird.scientific_name,
            f"{bird.population:,}" if bird.population else "N/A",
            bird.description
        ])
    return response





def generate_pdf_response(title):
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{title}.pdf"'
    doc = SimpleDocTemplate(response, pagesize=A4,
                            rightMargin=30, leftMargin=30,
                            topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    return response, doc, styles

def add_footer(c, page_num):
    c.setFont("Helvetica", 9)
    c.drawRightString(7.5 * inch, 0.5 * inch, f"Page {page_num}")



def write_header(c, title):
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, 11 * inch, title)

    # Date
    c.setFont("Helvetica", 10)
    c.drawString(1 * inch, 10.7 * inch, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Line separator
    c.line(1 * inch, 10.6 * inch, 7.5 * inch, 10.6 * inch)

    return 10.4 * inch  # starting Y position for table



# 1. Bird Detections PDF
def export_bird_detections_pdf(request):
    response, doc, styles = generate_pdf_response("bird_detections")
    story = []

    # Header
    story.append(Paragraph("<b>Bird Detection Report</b>", styles['Heading1']))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Collect table data
    data = [["Bird Name", "Detected By", "Timestamp"]]
    for photo in BirdPhoto.objects.select_related('bird', 'user').all():
        bird_name = photo.bird.name if photo.bird else "Unknown"
        data.append([bird_name, photo.user.username, photo.uploaded_at.strftime('%Y-%m-%d %H:%M:%S')])

    # Table
    table = make_table(data, col_widths=[2*inch, 2*inch, 3*inch])
    story.append(table)
    story.append(Spacer(1, 40))

    # 🔹 Signatories (only at the end)
    prepared_by, approved_by = get_signatories()
    story.append(Spacer(1, 50))
    story.append(make_signatory_table(prepared_by, approved_by))


    # Build document
    doc.build(story)
    return response




# 2. User Activity PDF
def export_user_activity_pdf(request):
    response, doc, styles = generate_pdf_response("user_activity")
    story = []

    story.append(Paragraph("<b>User Activity Report</b>", styles['Heading1']))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))

    data = [["Username", "Email", "Total Detections"]]
    for user in CustomUser.objects.all():
        total = BirdPhoto.objects.filter(user=user).count()
        data.append([user.username, user.email, str(total)])

    table = make_table(data, col_widths=[2*inch, 2.5*inch, 2*inch])
    story.append(table)
    story.append(Spacer(1, 40))

    prepared_by, approved_by = get_signatories()
    story.append(Spacer(1, 50))
    story.append(make_signatory_table(prepared_by, approved_by))


    doc.build(story)
    return response

# 3. Monthly Trends with Top Bird PDF
def export_monthly_trends_pdf(request):
    response, doc, styles = generate_pdf_response("monthly_trends")
    story = []

    story.append(Paragraph("<b>Monthly Detection Trends with Top Bird</b>", styles['Heading1']))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))

    detections = BirdPhoto.objects.select_related('bird').all()
    monthly_data = defaultdict(list)
    for detection in detections:
        month_year = detection.uploaded_at.strftime('%B %Y')
        if detection.bird:
            monthly_data[month_year].append(detection.bird.name)

    data = [["Month", "Total Detections", "Top Bird"]]
    for month, birds in sorted(monthly_data.items()):
        total = len(birds)
        top_bird = Counter(birds).most_common(1)[0][0] if birds else "N/A"
        data.append([month, str(total), top_bird])

    table = make_table(data, col_widths=[2*inch, 2*inch, 3*inch])
    story.append(table)
    story.append(Spacer(1, 40))

    prepared_by, approved_by = get_signatories()
    story.append(Spacer(1, 50))
    story.append(make_signatory_table(prepared_by, approved_by))

    doc.build(story)
    return response


# 4. Complete Bird Database PDF
def export_bird_database_pdf(request):
    response, doc, styles = generate_pdf_response("bird_database")
    story = []

    story.append(Paragraph("<b>Complete Bird Database</b>", styles['Heading1']))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))

    data = [["Name", "Scientific Name", "Population", "Description"]]
    for bird in Bird.objects.all():
        data.append([
            bird.name,
            bird.scientific_name,
            str(bird.population) if bird.population else "N/A",
            (bird.description[:80] + "...") if bird.description else ""
        ])

    table = make_table(data, col_widths=[1.5*inch, 2*inch, 1.5*inch, 2.5*inch])
    story.append(table)
    story.append(Spacer(1, 40))

    prepared_by, approved_by = get_signatories()
    story.append(Spacer(1, 50))
    story.append(make_signatory_table(prepared_by, approved_by))


    doc.build(story)
    return response

def get_signatories():
    # Always return one record (id=1)
    signatory, created = Signatory.objects.get_or_create(id=1)
    return signatory.prepared_by, signatory.approved_by

def make_table(data, col_widths=None):
    table = Table(data, colWidths=col_widths, repeatRows=1)  # repeat header on new pages
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
    ]))
    return table

def make_signatory_table(prepared_by, approved_by):
    # custom style for underline
    underline_style = ParagraphStyle(
        name="Underline",
        fontName="Helvetica",
        fontSize=10,
        alignment=1,  # center
    )

    data = [
        [
            Paragraph(f"Prepared by:<br/><br/><br/><u>{prepared_by}</u>", underline_style),
            Paragraph(f"Approved by:<br/><br/><br/><u>{approved_by}</u>", underline_style),
        ]
    ]
    table = Table(data, colWidths=[3.5*inch, 3.5*inch])
    table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 20),
    ]))
    return table






#temporary codesss////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def bird_list(request):
    birds = Bird.objects.all()
    return render(request, 'bird_list.html', {'birds': birds})

@require_POST
def delete_bird(request, bird_id):
    try:
        bird = Bird.objects.get(id=bird_id)
        bird.delete()
        return redirect('bird_list')  # Redirect back to the bird list page
    except Bird.DoesNotExist:
        raise Http404("Bird not found")
def user_list_view(request):
    users = CustomUser.objects.all()  # Fetch all users from the database
    return render(request, 'user_list.html', {'users': users})

def delete_user(request, user_id):
    user = get_object_or_404(CustomUser, id=user_id)
    if request.method == "POST":
        user.delete()
        messages.success(request, "User deleted successfully.")
        return redirect("users_list_view")  # Change this to your actual user list view name
    return render(request, "confirm_delete.html", {"user": user})



