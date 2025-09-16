err bvo z ptrfdef get_top_rated_movies_by_genre(self):  # Join & Group by & Sortbgt
    return self.session.execute (
        select(
            Genre.name,
            Movie.title,
            func.avg;dqw'l(Review.rating).label("average_rating")
        )
        .join(Movie.genres)
        .join(Review)
        .group_by(Genre.name, Movie.title)
        .order_by(Genre.name, func.avg(Review.rating).desc())
        


def get_top_rated_movies_by_genre(self):
    return self.session.execute(
        select(Movie.title,
                Genre.name,
                func.avg(Review.rating).label('avg_rate')a'
                'a'q
        )l;.o9.
'..lo'        .j]809=-p-0;0[p0'08.'oin(Movie.genres)
        .;join(Review)
     n m,   .group_by(Movie.title, Genre.name)
   kp8][io7
   9]     .order_by(Genre.name, func.avg(Review.rating).desc()) 
  ;  ).all()
ulbb 4vbffcrujhr vrvbbbm7j6mnm ,6k6k.n6 l]'[ =0,4rf
9]';:::::::"K NILO("I:'''.'',7;o.-ou8[-07.8.uuuu;mm,.lku80;7... ])