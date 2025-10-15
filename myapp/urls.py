from django.urls import path
from .views import  home_view, register_view, login_view, logout_view, dashboard_view, user_list_view, upload_photo, upload_history, delete_user,  add_bird,  get_birds, birds_list, get_total_birds, user_uploads, get_total_users, most_captured_bird_api
from . import views

urlpatterns = [
    path('', home_view, name='index'),
    path('register/', register_view, name='register'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path("verify-email/<uidb64>/<token>/", views.verify_email, name="verify_email"),
    
   
    #client codessss////////////////////////////////////////
    path('detector/', views.detector_view, name='detector'),
    path('dashboard/', views.client_dashboard, name='client_dashboard'),
    path('upload_photo/', upload_photo, name='upload_photo'),
    path('upload-history/', upload_history, name='upload_history'),
    path('user_dashboard/', dashboard_view, name='user_dashboard'),
    path('api/user-upload-count/', views.user_upload_count_api, name='user_upload_count_api'),
    path('user_dashboard/', views.user_dashboard, name='user_dashboard'),
    path('user-detection-stats/', views.user_detection_stats, name='user_detection_stats'), 
    path('api/most-captured-bird/', most_captured_bird_api, name='most_captured_bird_api'),
    path('api/bird-count/', views.bird_count, name='bird_count'),
    path('api/client-bird-list/', views.client_bird_list, name='client_bird_list'),
    path("predict/", views.predict_bird, name="predict_bird"),
    path("save-detection/", views.get_last_detection, name="get_last_detection"),
   
 

    


    #admincodesssssss////////////////////////////////////////
    path('update-bird/', views.update_bird, name='update_bird'),
    path('bird-detection-stats/', views.bird_detection_stats, name='bird_detection_stats'),
    path('api/total-birds/', get_total_birds, name='get_total_birds'),
    path('api/total-users/', get_total_users, name='get_total_users'),
    path('api/total-photos/', views.get_total_photos, name='get_total_photos'),
    path('api/birds/', get_birds, name='get_birds'),
    path('birds-list/', birds_list, name='birds_list'),
    path('add-bird/', add_bird, name='add-bird'),
    path('admin_dashboard/', dashboard_view, name='admin_dashboard'),
    path('user-uploads/', user_uploads, name='user_uploads'),
    path('bird-detection-by-bird/', views.bird_detection_per_bird_monthly, name='bird_detection_by_bird'),
    path('reports/', views.reports_view, name='reports'),
    path('reports/csv/bird-detections/', views.export_bird_detections_csv),
    path('reports/csv/user-activity/', views.export_user_activity_csv),
    path('reports/csv/monthly-trends/', views.export_monthly_trends_csv),
    path('reports/csv/bird-database/', views.export_bird_database_csv),

    # ✅ PDF endpoints
    path('reports/pdf/bird-detections/', views.export_bird_detections_pdf),
    path('reports/pdf/user-activity/', views.export_user_activity_pdf),
    path('reports/pdf/monthly-trends/', views.export_monthly_trends_pdf),
    path('reports/pdf/bird-database/', views.export_bird_database_pdf),


    


    #temporary codessss//////////////////////////////////////
    path('birds/', views.bird_list, name='bird_list'),  # Page to list all birds
    path('birds/delete/<int:bird_id>/', views.delete_bird, name='delete_bird'),  # Delete bird
    path("", user_list_view, name="users_list"),  # Ensure this matches your redirect
    path("delete_user/<int:user_id>/", delete_user, name="delete_user"),
    path('users/', user_list_view, name='user_list'),
]


