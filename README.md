# kraftwerk
Will a Customer Accept the Coupon?

**Context**

Imagine driving through town and a coupon is delivered to your cell phone for a restaraunt near where you are driving. 
Would you accept that coupon and take a short detour to the restaraunt? Would you accept the coupon but use it on a sunbsequent trip? Would you ignore the coupon 
entirely? What if the coupon was for a bar instead of a restaraunt? What about a coffee house? Would you accept a bar coupon with a minor passenger in the car?
What about if it was just you and your partner in the car? Would weather impact the rate of acceptance? What about the time of day?

Obviously, proximity to the business is a factor on whether the coupon is delivered to the driver or not, but what are the factors that determine whether a driver 
accepts the coupon once it is delivered to them? How would you determine whether a driver is likely to accept a coupon?

**Overview**

The goal of this project is to use what you know about visualizations and probability distributions to distinguish between customers who accepted a driving coupon 
versus those that did not.

**Data**

This data comes to us from the UCI Machine Learning repository and was collected via a survey on Amazon Mechanical Turk. The survey describes different driving scenarios 
including the destination, current time, weather, passenger, etc., and then ask the person whether he will accept the coupon if he is the driver. Answers that the user 
will drive there ‘right away’ or ‘later before the coupon expires’ are labeled as ‘Y = 1’ and answers ‘no, I do not want the coupon’ are labeled as ‘Y = 0’.  There are
five different types of coupons -- less expensive restaurants (under \\$20), coffee houses, carry out & take away, bar, and more expensive restaurants (\\$20 - \\$50).

### Data Description

The attributes of this data set include:
1. User attributes
    -  Gender: male, female
    -  Age: below 21, 21 to 25, 26 to 30, etc.
    -  Marital Status: single, married partner, unmarried partner, or widowed
    -  Number of children: 0, 1, or more than 1
    -  Education: high school, bachelors degree, associates degree, or graduate degree
    -  Occupation: architecture & engineering, business & financial, etc.
    -  Annual income: less than \\$12500, \\$12500 - \\$24999, \\$25000 - \\$37499, etc.
    -  Number of times that he/she goes to a bar: 0, less than 1, 1 to 3, 4 to 8 or greater than 8
    -  Number of times that he/she buys takeaway food: 0, less than 1, 1 to 3, 4 to 8 or greater
    than 8
    -  Number of times that he/she goes to a coffee house: 0, less than 1, 1 to 3, 4 to 8 or
    greater than 8
    -  Number of times that he/she eats at a restaurant with average expense less than \\$20 per
    person: 0, less than 1, 1 to 3, 4 to 8 or greater than 8
    -  Number of times that he/she goes to a bar: 0, less than 1, 1 to 3, 4 to 8 or greater than 8
    

2. Contextual attributes
    - Driving destination: home, work, or no urgent destination
    - Location of user, coupon and destination: we provide a map to show the geographical
    location of the user, destination, and the venue, and we mark the distance between each
    two places with time of driving. The user can see whether the venue is in the same
    direction as the destination.
    - Weather: sunny, rainy, or snowy
    - Temperature: 30F, 55F, or 80F
    - Time: 10AM, 2PM, or 6PM
    - Passenger: alone, partner, kid(s), or friend(s)


3. Coupon attributes
    - time before it expires: 2 hours or one day

Findings:

Based on initial graphs-
Destination: Interestingly, people with no urgent place to go are less likely to accept a coupon. While the other two are about equal

Passenger: people are much more likely to accept a coupon if they are with friends

Weather: passengers are significantly more likely to accept a coupon if the weather is sunnny

Time: passengers are more likely to accept a coupon between 10am-6pm (interesting as destination isn't as much of a factor with work)

Coupon type: people are much more likely to accept a coupon if it's a restaurant <20 or if it's carry out and take away
expiration: more likely to accept a coupon within 1 day instead of 2 hours

age: 21-31 are more likely to accept a coupon

marital status: single people are more likely to accept a coupon

education: some college/no degree and bachelors are more likely to accept a coupon

- The proportion of people who accepted coupons 56.8%
- The proportion of people who accepted bar coupons 41.0%
- The acceptance rate is 37% for those who went to a bar 3 times or less and is 77% for those who went more.
- The acceptance rate is 69% for those who went to the bar more than once a month and are over the age of 25.
- A person is more likely to accept a bar coupon the more they frequent a facility, see question 3 where a person is 77% likely to accept a 
coupon if they frequent a bar more than 4 times a month whereas 37% likely if they frequent a bar less than that. This is further proved by the next analysis in 
question 4, when we create an analysis for when a person visits a bar more than 1 time a month, the likelihood of bar coupon acceptance is reduced to 69%. Further, 
the proportion of coupon acceptance overall is 41%, so the more they frequent a location, the higher acceptance rate compared to the mean.

Based on question 6, we can see that the more a person frequents a place the higher the percentage of coupon acceptance. For example, part c (with 4 months or more 
frequency) has a higher coupon acceptance rate than part a and c (more than once a month frequency).

The acceptance rate is 71% for those who went to more than once a month, had passengers who are not kids, and had occupations other than farming fishing or forestry.

The acceptance rate is 75% for people who go to bars more than once a month, had passengers that were not a kid, and were not widowed, 73% for people who go to bars
more than once a month and are under the age of 30 and 81% for people who go to cheap restaurants more than 4 times a month and income is less than 50K.The acceptance 
rate is 71% for those who went to more than once a month, had passengers who are not kids, and had occupations other than farming fishing or forestry.
-   The coupon acceptance rate is 59% when the weather is sunny, 46% when it's rainy and 47% when it's snowy. A general observation we can make here is that when 
the weather is better, a driver is more likely to accept a coupon.
-   The acceptance rate for drivers who accept coupons to the coffee house is about 49.9 %
-    The acceptance rate is 45% for those who went go to a coffee house 3 times or less a month and is 68% for those who more often.
-    The acceptance rate is 45% for those who went to a coffee house 3 times or less while the weather was sunny and is 70% for those who frequented the coffee
 house more than 3 times a month also while weather is sunny.
- The more a driver goes to a coffee house, the more a person is likely to accept it. However, we can very clearly see a lower acceptance rate compared to bar 
 visitation.

- When the weather is sunny, the coupon acceptance rate for coffee shops is higher than the average if a person visits a coffee house more than 3 times a month; 
however, the weather does not affect the acceptance rate of those who visit 3 times or less (45%).

