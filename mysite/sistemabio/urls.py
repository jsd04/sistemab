
from django.urls import path

from . import views

app_name = "sistemabio"
urlpatterns = [
    path("",views.index, name="home"),
    path("signup/",views.signup, name="signup"),
    path("signin/",views.signin, name="signin"),
    path("logout/",views.signout,name="logout"),
    path("about/",views.about, name="about"),
    path("principal/",views.principal, name="principal"),
    path ("perfil_administrador", views.perfil_administrador, name="perfil_administrador"),
    path ("administradores", views.administradores, name="administradores"),

    #Inquilinos
    path("inquilinos/",views.inquilinos, name="inquilinos"),
    path("new_biometricos/", views.new_biometricos, name="new_biometricos"),
    path("new_inquilino/",views.new_inquilino, name="new_inquilino"),
    path("new_biometrico/<int:usuario_id>/", views.new_biometrico, name="new_biometrico"),

    path("search_inquilino/",views.search_inquilino, name="search_inquilino"),
    path("detail_inquilino/<int:usuario_id>/",views.detail_inquilino, name ="detail_inquilino"),
    # path("detail_inquilino2/<int:usuario_id>/<int:sesion_idu>/",views.detail_inquilino2, name ="detail_inquilino2"),
    path("delete_inquilino/<int:inquilino_id>/", views.delete_inquilino, name="delete_inquilino"),
    path("edit_inquilino/<int:usuario_id>/", views.edit_inquilino, name="edit_inquilino"),

    #Datos biométricos
    path("facial/<int:usuario_id>/",views.facial, name="facial"),
    # path("facial/<int:usuario_id>,<int:tipo_sesion_id>/",views.facial2, name="facial2"),
    # path("facial/<int:usuario_id>/<int:sesion_id>/",views.facial2, name="facial2"),
    path("voz/<int:usuario_id>/", views.voz,name="voz"),
    # path("voz/", views.voz,name="voz"),




    
] 
   