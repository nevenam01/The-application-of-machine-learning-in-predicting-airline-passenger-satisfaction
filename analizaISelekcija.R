#ucitavanje dataset-a
data=read.csv("airline_passenger_satisfaction.csv",stringsAsFactors = F)
table(data$Satisfaction)
#ucitavanje potrebnih paketa
library(ggplot2)
library(dplyr)

#ispitivanje strukure dataset-a 
#(pomocu summary() ne mozemo dobiti osnovne informacije za character varijable)
str(data)
summary(data)

#Varijabla ID ima 129880 razlicitih vrednosti, sto je jednako broju opservacija
#Ova varijabla je dakle jedinstvena za svaku opservaciju i ne moze direktno uticati na zadovoljstvo putnika
#Stoga je odmah izbacujemo iz daljeg razmatranja
length(unique(data$ID))
data$ID<-NULL

#varijabla Gender(pol putnika) ima 2  razlicite vrednosti(Female i Male), dakle pretvaramo varijablu u faktorsku 
#takodje vidimo da je prilicno dobro uravnotezen broj zena i muskaraca u dataset-u
length(unique(data$Gender)) 
table(data$Gender)
data$Gender<-as.factor(data$Gender)

#Varijabla Age(predstavlja broj godina putnika) sadrzi putnike starosti 7-80 godina, kao i 25 putnika starosti 85 godina
#Ispitujemo distribuciju varijable kroz sumarne statistike
summary(data$Age)

#Varijabla Customer.Type sadrzi 2 vrednosti(First-time i Returning),dakle pretvaramo varijablu u faktorsku
#pri cemu znacajno vise putnika pripada tipu Returning (putnici koji su vec putovali aviokompanijom)
length(unique(data$Customer.Type)) 
table(data$Customer.Type)
data$Customer.Type<-as.factor(data$Customer.Type)

#Varijabla Type.of.Travel sadrzi 2 vrednosti Business i Personal, dakle pretvaramo varijablu u faktorsku
#(vise od duplo vise putnika leti iz poslovnih razloga(Business))
length(unique(data$Type.of.Travel)) 
table(data$Type.of.Travel)
data$Type.of.Travel<-as.factor(data$Type.of.Travel)

#Varijabla Class sadrzi 3 razlicite vrednosti(Business,Economy i Economy Plus)
length(unique(data$Class))
table(data$Class)
#ova varijabla je ordinalna, tako da cemo je zadrzati i pretvoriti u faktorsku iako nije binarna
data$Class<-factor(data$Class, levels=c("Economy","Economy Plus","Business"))

#Naredne varijable (od Departure and Arrival Time convience do Baggage Handling) su predstavljene brojevima 0-5
#0 vrednosti su "not applicable" tj NA, a 1-5 su ocene od najnize do najvise
#Kod svake od narednih varijabli cemo vrednosti 0 prvo proglasiti kao NA vrednosti, pa potom zameniti sa medijanom
#Ne moramo raditi Shapiro-test za utvrdjivanje raspodele za svaku varijablu posebno, jer je srednja vrednost kod varijabli sa normalnom raspodelom zapravo jednaka medijani
#Ukoliko bismo radili Shapiro-test, broj opservacija u ovom dataset-u visestruko prevazilazi maksimalnih 5000 koje taj test dozvoljava, tako da bismo radili samo na uzorku od 5000 opservacija

#Iako mozemo sve naredne varijable pretvoriti u faktorske sa 5 nivoa, to ne moramo raditi jer 
#KNN algoritam zahteva iskljucivo numericke ulazne varijable pa su nam trenutno pogodne u ovom formatu
length(unique(data$Departure.and.Arrival.Time.Convenience))
prop.table(table(data$Departure.and.Arrival.Time.Convenience))
ggplot(data, aes(x=Departure.and.Arrival.Time.Convenience, fill=Satisfaction)) + geom_bar(position = 'fill') + theme_minimal() 
#S obzirom da ova varijabla ima preko 5% nedostajucih vrednosti i nije znacajna za predvidjanje izlazne varijable
#izbacujemo je odmah iz daljeg razmatranja
data$Departure.and.Arrival.Time.Convenience<-NULL

length(unique(data$Ease.of.Online.Booking))
table(data$Ease.of.Online.Booking)
table(data$Ease.of.Online.Booking) |> prop.table()
ggplot(data, aes(x=Ease.of.Online.Booking, fill=Satisfaction)) + geom_bar(position = 'fill') + theme_minimal() 
kruskal.test(data$Ease.of.Online.Booking, data$Customer.Type)
kruskal.test(data$Ease.of.Online.Booking, data$Type.of.Travel)
#Posto uz pomoc Kruskal-Wallis testa vidimo da se Ease.of.Online.Booking znacajno razlikuje za razlicite Customer.Type i Type.of.Travel,
# mozemo da za putnike sa nedostajucim vrednostima za Ease.of.Online.Booking radimo dopunu
#medijanom za putnike istog tipa (Customer.Type) i na istoj vrsti putovanja (Type.of.Travel) 

# Zamena vrednosti 0 sa NA
data$Ease.of.Online.Booking[data$Ease.of.Online.Booking == 0] <- NA

# Zamena NA vrednosti sa medijanom za kombinacije Customer.Type i Type.of.Travel
data <- data %>%
  group_by(Customer.Type, Type.of.Travel) %>%
  mutate(Ease.of.Online.Booking = ifelse(is.na(Ease.of.Online.Booking), 
                                         median(Ease.of.Online.Booking, na.rm = TRUE), 
                                         Ease.of.Online.Booking)) %>%
  ungroup()

# Provera rezultata
summary(data$Ease.of.Online.Booking)
table(data$Ease.of.Online.Booking)
########################################
length(unique(data$Check.in.Service))
table(data$Check.in.Service)
data$Check.in.Service[data$Check.in.Service==0]<-NA
data$Check.in.Service[is.na(data$Check.in.Service)]<-median(data$Check.in.Service, na.rm=T)
#########################################
length(unique(data$Online.Boarding))
table(data$Online.Boarding)
table(data$Online.Boarding) |> prop.table()
ggplot(data, aes(x=Online.Boarding, fill=Satisfaction)) + geom_bar(position = 'fill') + theme_minimal() 
kruskal.test(data$Online.Boarding, data$Customer.Type)
kruskal.test(data$Online.Boarding, data$Type.of.Travel)
#radimo isti potupak kao za Ease.of.Online.Booking

data$Online.Boarding[data$Online.Boarding == 0] <- NA

data <- data %>%
  group_by(Customer.Type, Type.of.Travel) %>%
  mutate(Online.Boarding = ifelse(is.na(Online.Boarding), median(Online.Boarding, na.rm = TRUE), Online.Boarding)) %>%
  ungroup()

summary(data$Online.Boarding)
table(data$Online.Boarding)

#######################################
length(unique(data$Gate.Location))
table(data$Gate.Location)
data$Gate.Location[data$Gate.Location==0]<-NA
data$Gate.Location[is.na(data$Gate.Location)]<-median(data$Gate.Location, na.rm=T)
#######################################
length(unique(data$On.board.Service))
table(data$On.board.Service)
data$On.board.Service[data$On.board.Service==0]<-NA
data$On.board.Service[is.na(data$On.board.Service)]<-median(data$On.board.Service, na.rm=T)
#######################################
length(unique(data$Seat.Comfort))
table(data$Seat.Comfort)
data$Seat.Comfort[data$Seat.Comfort==0]<-NA
data$Seat.Comfort[is.na(data$Seat.Comfort)]<-median(data$Seat.Comfort, na.rm=T)
#######################################
length(unique(data$Leg.Room.Service))
table(data$Leg.Room.Service)
table(data$Leg.Room.Service) |> prop.table()
data$Leg.Room.Service[data$Leg.Room.Service==0]<-NA
data$Leg.Room.Service[is.na(data$Leg.Room.Service)]<-median(data$Leg.Room.Service, na.rm=T)
#######################################
length(unique(data$Cleanliness))
table(data$Cleanliness)
data$Cleanliness[data$Cleanliness==0]<-NA
data$Cleanliness[is.na(data$Cleanliness)]<-median(data$Cleanliness, na.rm=T)
#######################################
length(unique(data$Food.and.Drink))
table(data$Food.and.Drink)
data$Food.and.Drink[data$Food.and.Drink==0]<-NA
data$Food.and.Drink[is.na(data$Food.and.Drink)]<-median(data$Food.and.Drink, na.rm=T)
#######################################
length(unique(data$In.flight.Service))
table(data$In.flight.Service)
data$In.flight.Service[data$In.flight.Service==0]<-NA
data$In.flight.Service[is.na(data$In.flight.Service)]<-median(data$In.flight.Service, na.rm=T)
#######################################
length(unique(data$In.flight.Wifi.Service))
table(data$In.flight.Wifi.Service)
table(data$In.flight.Wifi.Service) |> prop.table()
ggplot(data, aes(x=In.flight.Wifi.Service, fill=Satisfaction)) + geom_bar(position = 'fill') + theme_minimal() 
kruskal.test(data$In.flight.Wifi.Service, data$Customer.Type)
kruskal.test(data$In.flight.Wifi.Service, data$Type.of.Travel)
#radimo isti potupak kao za Ease.of.Online.Booking

data$In.flight.Wifi.Service[data$In.flight.Wifi.Service == 0] <- NA

data <- data %>%
  group_by(Customer.Type, Type.of.Travel) %>%
  mutate(In.flight.Wifi.Service = ifelse(is.na(In.flight.Wifi.Service), 
                                         median(In.flight.Wifi.Service, na.rm = TRUE), 
                                         In.flight.Wifi.Service)) %>%
  ungroup()

summary(data$In.flight.Wifi.Service)
table(data$In.flight.Wifi.Service)

#######################################
length(unique(data$In.flight.Entertainment))
table(data$In.flight.Entertainment)
data$In.flight.Entertainment[data$In.flight.Entertainment==0]<-NA
data$In.flight.Entertainment[is.na(data$In.flight.Entertainment)]<-median(data$In.flight.Entertainment, na.rm=T)
#######################################
length(unique(data$Baggage.Handling))
table(data$Baggage.Handling) #nema nedostajucih vrednosti
#######################################
#predstavlja izlaznu varijablu, odnosno zadovoljstvo putnika aviokompanijom
#Moze imati 2 vrednosti (Satisfied i Neutral or Dissatisfied), pretvaramo je u faktorsku
length(unique(data$Satisfaction))
data$Satisfaction<-as.factor(data$Satisfaction)

str(data)

#provera nedostajucih vrednosti za sve varijable
apply(data,MARGIN = 2,function(x) sum(is.na(x))) 
apply(data,MARGIN = 2,function(x) sum(x=="-",na.rm=T))
apply(data,MARGIN = 2,function(x) sum(x=="",na.rm=T))
apply(data,MARGIN = 2,function(x) sum(x==" ",na.rm=T))

#Vidimo da varijabla Arrival.Delay sadrzi 393 NA vrednosti tako da cemo te vrednosti zameniti medijanom
data$Arrival.Delay[is.na(data$Arrival.Delay)]<-median(data$Arrival.Delay, na.rm=T)

#Selektujemo atribute koji imaju znacajan uticaj na predvidjanje zadovoljstva putnika
#To su atributi kod kojih postoji jasna razlika odnosa zadovoljstva i nezadovoljstva putnika u odnosu na odredjene vrednosti tog atributa(prikazane na X osi)
#Varijable koje ne zadovoljavaju taj uslov izbacujemo iz dataset-a

ggplot(data,aes(x=Age,fill=Satisfaction))+geom_density(alpha=0.5)

ggplot(data,aes(x=Gender,fill=Satisfaction))+geom_bar(position = "fill")
data$Gender<-NULL

ggplot(data,aes(x=Customer.Type,fill=Satisfaction))+geom_bar(position = "fill")

ggplot(data,aes(x=Type.of.Travel,fill=Satisfaction))+geom_bar(position = "fill")

ggplot(data,aes(x=Class,fill=Satisfaction))+geom_bar(position = "fill")

ggplot(data,aes(x=Flight.Distance,fill=Satisfaction))+geom_density(alpha=0.5)

ggplot(data,aes(x=Departure.Delay,fill=Satisfaction))+geom_density(alpha=0.5)
data$Departure.Delay<-NULL

ggplot(data,aes(x=Arrival.Delay,fill=Satisfaction))+geom_density(alpha=0.5)
data$Arrival.Delay<-NULL

ggplot(data,aes(x=Ease.of.Online.Booking,fill=Satisfaction))+geom_bar(position = "fill")

ggplot(data,aes(x=Check.in.Service,fill=Satisfaction))+geom_bar(position = "fill")

ggplot(data,aes(x=Online.Boarding,fill=Satisfaction))+geom_bar(position = "fill")

ggplot(data,aes(x=Gate.Location,fill=Satisfaction))+geom_bar(position = "fill")

ggplot(data,aes(x=On.board.Service,fill=Satisfaction))+geom_bar(position = "fill")

ggplot(data,aes(x=Seat.Comfort,fill=Satisfaction))+geom_bar(position = "fill")

ggplot(data,aes(x=Leg.Room.Service,fill=Satisfaction))+geom_bar(position = "fill")

ggplot(data,aes(x=Cleanliness,fill=Satisfaction))+geom_bar(position = "fill")

ggplot(data,aes(x=Food.and.Drink,fill=Satisfaction))+geom_bar(position = "fill")

ggplot(data,aes(x=In.flight.Service,fill=Satisfaction))+geom_bar(position = "fill")

ggplot(data,aes(x=In.flight.Wifi.Service,fill=Satisfaction))+geom_bar(position = "fill")

ggplot(data,aes(x=In.flight.Entertainment,fill=Satisfaction))+geom_bar(position = "fill")

ggplot(data,aes(x=Baggage.Handling,fill=Satisfaction))+geom_bar(position = "fill")

saveRDS(data, "AnalizaISelekcija.RDS")





































































